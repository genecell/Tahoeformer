# datasets.py

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import pyfaidx 
import kipoiseq.transforms.functional 
from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.Chem import rdFingerprintGenerator

# --- Global Config ---
# Enformer typically uses a 196,608 bp input sequence.
# We will use a shorter input (1/4 of usual length) to speed up training.
ENFORMER_INPUT_SEQ_LENGTH = 49_152 

# Relative paths from the project root directory
GENOME_FASTA_PATH = "data/hg38.fa"
TSS_REGIONS_CSV_PATH = "data/Enformer_genomic_regions_TSSCenteredGenes_FixedOverlapRemoval.csv"

# Path to pseudobulk target data, matching the provided dummy file
PSEUDOBULK_TARGET_DATA_PATH = "data/pseudobulk_dummy.csv"

# ----------------------


class GenomeOneHotEncoder:
    """
    Encodes DNA sequences into one-hot format using kipoiseq.
    """
    def __init__(self, sequence_length: int = ENFORMER_INPUT_SEQ_LENGTH):
        self.sequence_length = sequence_length

    @staticmethod
    def _one_hot_encode(sequence: str) -> np.ndarray:
        ## one hot encodes DNA using the same code from the original Enformer paper. 
        ## Ensures one-hot encoding is consistent with representations Enformer has 
        ## already learned
        return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)

    def encode(self, seq: str) -> np.ndarray:
        """
        One-hot encodes a DNA sequence using kipoiseq.

        Args:
            seq (str): The DNA sequence string. The FastaReader should ensure this
                       sequence is already uppercase and of length ENFORMER_INPUT_SEQ_LENGTH.

        Returns:
            np.ndarray: A numpy array, typically (L, 4) for DNA, with one-hot encoded sequence.
        """

        return GenomeOneHotEncoder._one_hot_encode(seq)


class FastaReader:
    """
    Reads sequences from a FASTA file using pyfaidx.
    Handles chromosome boundary conditions by padding with 'N'.
    """
    def __init__(self, fasta_path: str):
        self.fasta_path = fasta_path
        self.genome = None
        try:
            self.genome = pyfaidx.Fasta(self.fasta_path, sequence_always_upper=True)
            print(f"Successfully loaded and indexed genome using pyfaidx from: {self.fasta_path}")
        except pyfaidx.FastaIndexingError as e:
            print(f"Error: Could not index FASTA file at {self.fasta_path}.")
            print("Ensure it's a valid FASTA file and the .fai index can be created/read in its directory.")
            print(f"pyfaidx error: {e}")
            raise
        except FileNotFoundError:
            print(f"Error: FASTA file not found at {self.fasta_path}.")
            raise

    def get_sequence(self, chrom: str, start_0based: int, end_0based_exclusive: int) -> str:
        """
        Fetches a DNA sequence for the given 0-based genomic interval.
        Pads with 'N' if the interval extends beyond chromosome boundaries.

        Args:
            chrom (str): Chromosome name (e.g., 'chr1').
            start_0based (int): 0-based start coordinate (inclusive).
            end_0based_exclusive (int): 0-based end coordinate (exclusive).

        Returns:
            str: The DNA sequence, padded with 'N's to match the requested length
                 (end_0based_exclusive - start_0based).
        """
        if self.genome is None:
            raise RuntimeError("FastaReader not properly initialized (pyfaidx missing or genome loading failed).")

        # Sanitize chromosome name (e.g., '1' vs 'chr1')
        true_chrom_name = chrom
        if chrom not in self.genome:
            alternative_chrom_name = 'chr' + chrom if not chrom.startswith('chr') else chrom.replace('chr', '', 1)
            if alternative_chrom_name in self.genome:
                true_chrom_name = alternative_chrom_name
            else:
                available_chroms_sample = list(self.genome.keys())[:5]
                raise ValueError(
                    f"Chromosome '{chrom}' (and alternative '{alternative_chrom_name}') not found in FASTA file. "
                    f"Available chromosomes sample: {available_chroms_sample}..."
                )
        
        chrom_len = len(self.genome[true_chrom_name])
        seq_len_requested = end_0based_exclusive - start_0based

        # init sequence with Ns for padding
        sequence_parts = []
        
        # handle padding at the beginning
        padding_start_len = 0
        if start_0based < 0:
            padding_start_len = abs(start_0based)
            sequence_parts.append('N' * padding_start_len)
            effective_start = 0
        else:
            effective_start = start_0based

        # determine the part of the sequence to fetch from FASTA
        fetch_len = min(end_0based_exclusive, chrom_len) - effective_start
        
        if fetch_len > 0:
            sequence_parts.append(self.genome[true_chrom_name][effective_start : effective_start + fetch_len].seq)
        elif effective_start >= chrom_len: # Requested start is beyond chromosome end
             pass # No sequence to fetch, only padding needed

        # handle padding at the end
        current_len = sum(len(p) for p in sequence_parts)
        padding_end_len = seq_len_requested - current_len
        if padding_end_len > 0:
            sequence_parts.append('N' * padding_end_len)
        
        final_sequence = "".join(sequence_parts)

        # Final check for length; this should be guaranteed by logic above
        if len(final_sequence) != seq_len_requested:
            # This indicates a logic error in padding/fetching
            raise RuntimeError(
                f"Internal error: Final sequence length {len(final_sequence)} for {true_chrom_name}:{start_0based}-{end_0based_exclusive} "
                f"does not match requested {seq_len_requested}."
            )
        return final_sequence


# --- Main Dataset Classes ---

class TahoeDataset(Dataset):
    """
    PyTorch Dataset for loading Tahoe data for Enformer fine-tuning.
    - Reads genomic regions from a regions CSV.
    - Reads pseudobulk conditions and expression values from a pseudobulk CSV.
    - Merges these two data sources based on gene identifiers.
    - Each sample represents a unique gene-condition pair.
    - Fetches DNA sequence for the gene, resized to `enformer_input_seq_length`.
    - One-hot encodes sequence and returns it with the specific expression value.
    """
    ORIGINAL_ENFORMER_WINDOW_SIZE = 196_608

    def __init__(self,
                 tss_regions_csv_path: str,
                 genome_fasta_path: str,
                 pseudobulk_data_path: str,
                 enformer_input_seq_length: int = ENFORMER_INPUT_SEQ_LENGTH,
                 regions_csv_gene_col: str = 'gene_name',        # Gene ID column in tss_regions_csv
                 pseudobulk_csv_gene_col: str = 'gene_id',     # Gene ID column in pseudobulk_data_csv
                 regions_csv_chr_col: str = 'seqnames',      # Chromosome column in tss_regions_csv
                 regions_csv_start_col: str = 'starts',        # 0-based start col in tss_regions_csv
                 regions_csv_end_col: str = 'ends'):           # 0-based exclusive end col in tss_regions_csv
        super().__init__()

        self.enformer_input_seq_length = enformer_input_seq_length
        # Store column names for clarity
        self.regions_gene_col = regions_csv_gene_col
        self.pseudobulk_gene_col = pseudobulk_csv_gene_col
        self.regions_chr_col = regions_csv_chr_col
        self.regions_start_col = regions_csv_start_col
        self.regions_end_col = regions_csv_end_col

        print(f"Initializing TahoeDataset...")
        print(f"  Target model input sequence length: {self.enformer_input_seq_length} bp")
        print(f"  Genomic regions are assumed to define a {self.ORIGINAL_ENFORMER_WINDOW_SIZE} bp window for centering.")

        # Load genomic regions data
        print(f"  Loading TSS regions from: {tss_regions_csv_path}")
        try:
            regions_df = pd.read_csv(tss_regions_csv_path)
            print(f"    Successfully loaded regions CSV with {len(regions_df)} gene region entries.")
            expected_region_cols = [self.regions_chr_col, self.regions_gene_col, 
                                    self.regions_start_col, self.regions_end_col]
            missing_region_cols = [col for col in expected_region_cols if col not in regions_df.columns]
            if missing_region_cols:
                raise ValueError(f"Missing columns in regions CSV ('{tss_regions_csv_path}'): {missing_region_cols}. Expected: {expected_region_cols}")
        except FileNotFoundError:
            print(f"FATAL ERROR: Regions CSV file not found at {tss_regions_csv_path}")
            raise
        except Exception as e:
            print(f"FATAL ERROR loading or validating regions CSV: {e}")
            raise

        # Load pseudobulk target data
        print(f"  Loading pseudobulk targets from: {pseudobulk_data_path}")
        try:
            pseudobulk_df = pd.read_csv(pseudobulk_data_path)
            print(f"    Successfully loaded pseudobulk CSV with {len(pseudobulk_df)} condition entries.")
            expected_pb_cols = [self.pseudobulk_gene_col, 'cell_line', 'drug_id', 'drug_dose', 'expression']
            missing_pb_cols = [col for col in expected_pb_cols if col not in pseudobulk_df.columns]
            if missing_pb_cols:
                raise ValueError(f"Missing columns in pseudobulk CSV ('{pseudobulk_data_path}'): {missing_pb_cols}. Expected: {expected_pb_cols}")
        except FileNotFoundError:
            print(f"FATAL ERROR: Pseudobulk CSV file not found at {pseudobulk_data_path}")
            raise
        except Exception as e:
            print(f"FATAL ERROR loading or validating pseudobulk CSV: {e}")
            raise

        # Merge regions with pseudobulk data
        print(f"  Merging genomic regions with pseudobulk target data...")
        print(f"    Regions gene column: '{self.regions_gene_col}', Pseudobulk gene column: '{self.pseudobulk_gene_col}'")
        
        regions_df[self.regions_gene_col] = regions_df[self.regions_gene_col].astype(str)
        pseudobulk_df[self.pseudobulk_gene_col] = pseudobulk_df[self.pseudobulk_gene_col].astype(str)

        self.samples_df = pd.merge(
            regions_df,
            pseudobulk_df,
            left_on=self.regions_gene_col,
            right_on=self.pseudobulk_gene_col,
            how='inner' # Keeps only genes present in both DataFrames
        )

        if len(self.samples_df) == 0:
            print("WARNING: The merge operation resulted in an empty DataFrame.")
            print(f"  No common genes found between column '{self.regions_gene_col}' in regions CSV ")
            print(f"  and column '{self.pseudobulk_gene_col}' in pseudobulk CSV.")
            print(f"  Please check that gene identifiers match and are of the same type in both files.")
            # Example gene IDs for debugging:
            if not regions_df.empty: print(f"  Sample gene IDs from regions CSV: {regions_df[self.regions_gene_col].unique()[:5].tolist()}")
            if not pseudobulk_df.empty: print(f"  Sample gene IDs from pseudobulk CSV: {pseudobulk_df[self.pseudobulk_gene_col].unique()[:5].tolist()}")
        else:
            print(f"    Successfully merged data: {len(self.samples_df)} total samples (gene-condition pairs).")
            
            # Check for genes in regions_df not found in pseudobulk_df (and thus dropped)
            original_region_genes = set(regions_df[self.regions_gene_col].unique())
            merged_region_genes = set(self.samples_df[self.regions_gene_col].unique())
            dropped_region_genes = original_region_genes - merged_region_genes
            if dropped_region_genes:
                print(f"    WARNING: {len(dropped_region_genes)} unique gene IDs from the regions CSV ('{self.regions_gene_col}') were not found in the pseudobulk CSV ('{self.pseudobulk_gene_col}') and were dropped.")
                print(f"      Examples of dropped region gene IDs: {list(dropped_region_genes)[:min(5, len(dropped_region_genes))]}")

            # Check for genes in pseudobulk_df not found in regions_df (and thus dropped)
            original_pseudobulk_genes = set(pseudobulk_df[self.pseudobulk_gene_col].unique())
            
            merged_pseudobulk_genes = set(self.samples_df[self.regions_gene_col].unique()) # Genes that made it into the merge, identified by the regions_gene_col key
            
            final_merged_keys_from_pseudobulk_perspective = set(self.samples_df[self.pseudobulk_gene_col].unique())
            dropped_pseudobulk_genes = original_pseudobulk_genes - final_merged_keys_from_pseudobulk_perspective
            
            if dropped_pseudobulk_genes:
                print(f"    WARNING: {len(dropped_pseudobulk_genes)} unique gene IDs from the pseudobulk CSV ('{self.pseudobulk_gene_col}') were not found in the regions CSV ('{self.regions_gene_col}') and were dropped.")
                print(f"      Examples of dropped pseudobulk gene IDs: {list(dropped_pseudobulk_genes)[:min(5, len(dropped_pseudobulk_genes))]}")

        if 'expression' in self.samples_df and self.samples_df['expression'].isnull().any():
             print("WARNING: NA values found in 'expression' column after merge. These samples might cause errors or yield NaN targets.")
             print("         Consider handling these (e.g., fill with a default or drop rows withna(subset=['expression'])).")
             # self.samples_df.dropna(subset=['expression'], inplace=True) # Example: drop rows with NA expression

        print(f"  Initializing FASTA reader for genome: {genome_fasta_path}")
        self.fasta_reader = FastaReader(genome_fasta_path)
        self.encoder = GenomeOneHotEncoder(sequence_length=self.enformer_input_seq_length)
        print("TahoeDataset initialized successfully.")

    def __len__(self):
        return len(self.samples_df)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if not (0 <= idx < len(self.samples_df)):
            raise IndexError(f"Index {idx} out of bounds for dataset of length {len(self.samples_df)}")

        sample_info = self.samples_df.iloc[idx]

        try:
            chrom = str(sample_info[self.regions_chr_col])
            # Gene name from the regions CSV (used for merge, should be consistent)
            gene_name_for_logging = str(sample_info[self.regions_gene_col]) 
            
            csv_region_start = int(sample_info[self.regions_start_col])
            csv_region_end = int(sample_info[self.regions_end_col])
            
            expression_value = float(sample_info['expression']) # Assuming 'expression' is the target column
        except KeyError as e:
            print(f"FATAL ERROR in __getitem__ (idx {idx}): Missing expected column {e} in merged samples_df.")
            print(f"  Available columns: {self.samples_df.columns.tolist()}")
            print(f"  Sample info for this index: {sample_info.to_dict() if isinstance(sample_info, pd.Series) else sample_info}")
            raise
        except ValueError as e:
            print(f"FATAL ERROR in __getitem__ (idx {idx}): Could not convert data for gene {gene_name_for_logging}. Error: {e}")
            print(f"  Expression value was: '{sample_info.get('expression', 'N/A')}'")
            raise
        except Exception as e: # Catch any other unexpected error for this item
            print(f"FATAL ERROR in __getitem__ (idx {idx}) for gene {gene_name_for_logging}: An unexpected error occurred: {type(e).__name__} - {e}")
            raise

        # --- Sequence window calculation ---
        actual_csv_window_len = csv_region_end - csv_region_start
        if actual_csv_window_len != self.ORIGINAL_ENFORMER_WINDOW_SIZE:
            # Warning if the input CSV regions are not consistently 196kb.
            # The centering logic below will still try to work based on csv_region_end.
            print(f"WARNING for gene {gene_name_for_logging} (idx {idx}): Region {chrom}:{csv_region_start}-{csv_region_end} from CSV "
                  f"has length {actual_csv_window_len}bp, but expected {self.ORIGINAL_ENFORMER_WINDOW_SIZE}bp "
                  f"for the original window definition used for centering. Sequence extraction might be affected if assumptions are wrong.")

        # Initialize final sequence coordinates with those from the CSV.
        # These will be used if no resizing is needed.
        final_seq_start_0based = csv_region_start
        final_seq_end_0based_exclusive = csv_region_end
        
        # If the target model input sequence length is different from the original Enformer window size,
        # recalculate start and end positions by centering the target length within the original window.
        if self.enformer_input_seq_length != self.ORIGINAL_ENFORMER_WINDOW_SIZE:
            # Calculate the center of the ORIGINAL_ENFORMER_WINDOW_SIZE.
            # Assumes 'csv_region_end' is the exclusive end of this original window.
            original_window_center = csv_region_end - (self.ORIGINAL_ENFORMER_WINDOW_SIZE // 2)
            
            half_target_seq_len = self.enformer_input_seq_length // 2
            final_seq_start_0based = original_window_center - half_target_seq_len
            # Ensure the end is exclusive and maintains the correct length for the target sequence
            final_seq_end_0based_exclusive = final_seq_start_0based + self.enformer_input_seq_length
        
        # Fetch and encode DNA sequence
        dna_sequence = self.fasta_reader.get_sequence(chrom, final_seq_start_0based, final_seq_end_0based_exclusive)
        one_hot_sequence = self.encoder.encode(dna_sequence) 
        one_hot_sequence_tensor = torch.tensor(one_hot_sequence, dtype=torch.float32)
        
        # Target is the specific expression value for this gene-condition pair
        target_tensor = torch.tensor([expression_value], dtype=torch.float32)

        return one_hot_sequence_tensor, target_tensor


# --- Extended Dataset for SMILES ---
class TahoeSMILESDataset(Dataset):
    """
    Extends TahoeDataset to also return:
        - Morgan Fingerprints for the drug
        - drug dose
        - target expression
    """
    def __init__(self,
                 regions_csv_path: str, # Renamed from tss_regions_csv_path for clarity with config
                 pbulk_parquet_path: str, # Renamed from pseudobulk_data_path for clarity with config
                 drug_meta_csv_path: str, # Renamed from drug_metadata_path for clarity with config
                 fasta_file_path: str,    # Renamed from genome_fasta_path for clarity with config
                 enformer_input_seq_length: int = ENFORMER_INPUT_SEQ_LENGTH,
                 # Morgan fingerprint parameters (from data_config)
                 morgan_fp_radius: int = 2,
                 morgan_fp_nbits: int = 2048,
                 # Column names from regions_csv (from data_config)
                 regions_gene_col: str    = 'gene_name',
                 regions_chr_col: str     = 'seqnames',
                 regions_start_col: str   = 'starts',
                 regions_end_col: str     = 'ends',
                 # Column names from pbulk_parquet (from data_config)
                 pbulk_gene_col: str      = 'gene_id',
                 pbulk_drug_col: str      = 'drug_id',
                 pbulk_dose_col: str      = 'drug_dose',
                 pbulk_expr_col: str      = 'expression',
                 pbulk_cell_line_col: str = 'cell_line',
                 # Column names from drug_meta_csv (from data_config)
                 drug_meta_id_col: str    = 'drug',
                 drug_meta_smiles_col: str = 'canonical_smiles',
                 filter_drugs_by_ids: list = None, # Added from dataset_args
                 regions_strand_col: str = None,    # Added from dataset_args, though not used in current __getitem__
                 regions_set_col: str = 'set',      # New: Name of the column in regions_csv for data splitting
                 target_set: str = None             # New: Specific set to load (e.g., "train", "valid", "test")
                 ):
        super().__init__()

        # store config
        self.seq_len = enformer_input_seq_length
        self.morgan_fp_radius = morgan_fp_radius
        self.morgan_fp_nbits = morgan_fp_nbits

        self.regions_gene_col    = regions_gene_col
        self.regions_chr_col     = regions_chr_col
        self.regions_start_col   = regions_start_col
        self.regions_end_col     = regions_end_col
        self.regions_set_col     = regions_set_col # Store the name of the set column

        self.pbulk_gene_col      = pbulk_gene_col
        self.pbulk_drug_col      = pbulk_drug_col
        self.pbulk_dose_col      = pbulk_dose_col
        self.pbulk_expr_col      = pbulk_expr_col
        self.pbulk_cell_line_col = pbulk_cell_line_col

        self.drug_meta_id_col    = drug_meta_id_col
        self.drug_meta_smiles_col= drug_meta_smiles_col
        
        self.target_set          = target_set # Store the specific set value for this instance

        # --- Morgan Fingerprint Generator (NEW) ---
        self._morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.morgan_fp_radius,
            fpSize=self.morgan_fp_nbits
        )

        # load & merge regions + pseudobulk
        print(f"  Loading TSS regions from: {regions_csv_path}")
        try:
            regs = pd.read_csv(regions_csv_path)
            print(f"    Successfully loaded regions CSV with {len(regs)} gene region entries.")
        except FileNotFoundError:
            print(f"FATAL ERROR: Regions CSV file not found at {regions_csv_path}")
            raise
        except Exception as e:
            print(f"FATAL ERROR loading regions CSV: {e}")
            raise

        print(f"  Loading pseudobulk targets from: {pbulk_parquet_path} (expected Parquet format)")
        try:
            pb = pd.read_parquet(pbulk_parquet_path)
            print(f"    Successfully loaded pseudobulk Parquet file with {len(pb)} entries.")
        except FileNotFoundError:
            print(f"FATAL ERROR: Pseudobulk Parquet file not found at {pbulk_parquet_path}")
            raise
        except Exception as e:
            print(f"FATAL ERROR loading or parsing pseudobulk Parquet file: {e}")
            print("  Ensure the file is a valid Parquet file and you have a Parquet engine like 'pyarrow' or 'fastparquet' installed.")
            raise

        # Ensure gene ID columns are strings for merging
        regs[self.regions_gene_col] = regs[self.regions_gene_col].astype(str)
        pb[self.pbulk_gene_col]     = pb[self.pbulk_gene_col].astype(str)

        print(f"  Merging genomic regions with pseudobulk target data...")
        print(f"    Regions gene column: '{self.regions_gene_col}', Pseudobulk gene column: '{self.pbulk_gene_col}'")
        self.samples_df = regs.merge(
            pb,
            left_on  = self.regions_gene_col,
            right_on = self.pbulk_gene_col,
            how      = 'inner'
        )
        
        if filter_drugs_by_ids and self.pbulk_drug_col in self.samples_df.columns:
            print(f"    Filtering samples to include only drugs: {filter_drugs_by_ids}")
            initial_count = len(self.samples_df)
            self.samples_df = self.samples_df[self.samples_df[self.pbulk_drug_col].isin(filter_drugs_by_ids)]
            print(f"    Retained {len(self.samples_df)} samples after drug filtering (from {initial_count}).")
            if len(self.samples_df) == 0:
                print("WARNING: No samples remaining after filtering by drug IDs. Check your filter_drugs_by_ids list and drug IDs in pbulk data.")

        # Filter by target_set if specified
        if self.target_set:
            if self.regions_set_col in self.samples_df.columns:
                print(f"    Filtering samples for set: '{self.target_set}' using column '{self.regions_set_col}'.")
                initial_count_set_filter = len(self.samples_df)
                self.samples_df = self.samples_df[self.samples_df[self.regions_set_col] == self.target_set].copy()
                print(f"    Retained {len(self.samples_df)} samples after filtering for set '{self.target_set}' (from {initial_count_set_filter}).")
                if len(self.samples_df) == 0:
                    print(f"WARNING: No samples remaining for this dataset instance (target_set='{self.target_set}') after filtering. Check the '{self.regions_set_col}' column in '{regions_csv_path}' for entries matching '{self.target_set}' and their overlap with pseudobulk data.")
            else:
                print(f"WARNING: target_set '{self.target_set}' was specified, but the column '{self.regions_set_col}' was not found in the merged DataFrame. No set-specific filtering was applied for this dataset instance. This instance will contain all data that matched other criteria.")

        # load drug metadata
        print(f"  Loading drug metadata from: {drug_meta_csv_path}")
        try:
            dm = pd.read_csv(drug_meta_csv_path)
            print(f"    Successfully loaded drug metadata with {len(dm)} entries.")
        except FileNotFoundError:
            print(f"FATAL ERROR: Drug metadata CSV not found at {drug_meta_csv_path}")
            raise
        except Exception as e:
            print(f"FATAL ERROR loading drug metadata CSV: {e}")
            raise
        
        # Ensure SMILES and ID columns are present and fill NA SMILES with empty string
        if self.drug_meta_smiles_col not in dm.columns:
            raise ValueError(f"SMILES column '{self.drug_meta_smiles_col}' not found in drug metadata.")
        if self.drug_meta_id_col not in dm.columns:
            raise ValueError(f"Drug ID column '{self.drug_meta_id_col}' not found in drug metadata.")
        dm[self.drug_meta_smiles_col] = dm[self.drug_meta_smiles_col].fillna('').astype(str)
        self.drug_meta = dm.set_index(self.drug_meta_id_col)

        # fasta reader & one-hot encoder
        self.fasta_reader = FastaReader(fasta_file_path)
        self.encoder      = GenomeOneHotEncoder(sequence_length=self.seq_len)
        print("TahoeSMILESDataset initialized.")

    def _generate_morgan_fingerprint(self, smiles_string: str) -> np.ndarray:
        """Generates a Morgan fingerprint from a SMILES string using the new generator API."""
        if not smiles_string:
            return np.zeros(self.morgan_fp_nbits, dtype=np.float32)
        try:
            mol = Chem.MolFromSmiles(smiles_string)
            if mol:
                # Use the generator's NumPy helper:
                fp_array = self._morgan_gen.GetFingerprintAsNumPy(mol)
                return fp_array.astype(np.float32)
            else:
                return np.zeros(self.morgan_fp_nbits, dtype=np.float32)
        except Exception as e:
            return np.zeros(self.morgan_fp_nbits, dtype=np.float32)

    def __len__(self):
        return len(self.samples_df)

    def __getitem__(self, idx):
        row = self.samples_df.iloc[idx]

        # --- DNA sequence ---
        chrom = str(row[self.regions_chr_col])
        start = int(row[self.regions_start_col])
        end   = int(row[self.regions_end_col])
        orig = end - start
        if self.seq_len != orig:
            center = end - orig//2
            half   = self.seq_len//2
            start, end = center-half, center+half

        seq = self.fasta_reader.get_sequence(chrom, start, end)
        oh  = self.encoder.encode(seq)
        seq_tensor = torch.tensor(oh, dtype=torch.float32)

        # --- Morgan Fingerprint --- 
        drug_id_for_fp = row[self.pbulk_drug_col]
        smiles_string = ''
        if drug_id_for_fp in self.drug_meta.index:
            smiles_string = self.drug_meta.loc[drug_id_for_fp, self.drug_meta_smiles_col]
            # If multiple entries for a drug_id, loc might return a Series. Take the first one.
            if isinstance(smiles_string, pd.Series):
                smiles_string = smiles_string.iloc[0]
        else:
            # print(f"Warning: Drug ID {drug_id_for_fp} not found in drug_meta. Using empty SMILES for fingerprint.")
            pass # SMILES string remains empty, will result in zero vector
        
        morgan_fp = self._generate_morgan_fingerprint(str(smiles_string)) # Ensure it's a string
        morgan_fp_tensor = torch.tensor(morgan_fp, dtype=torch.float32)

        # --- dose & target ---
        dose_val = float(row[self.pbulk_dose_col])
        expression_val = float(row[self.pbulk_expr_col])

        dose_tensor = torch.tensor([dose_val], dtype=torch.float32)
        tgt_tensor  = torch.tensor([expression_val], dtype=torch.float32)

        # --- Metadata for Logging ---
        gene_id_meta = str(row[self.pbulk_gene_col]) 
        drug_id_meta = str(row[self.pbulk_drug_col])
        cell_line_meta = str(row[self.pbulk_cell_line_col])

        return seq_tensor, morgan_fp_tensor, dose_tensor, tgt_tensor, gene_id_meta, drug_id_meta, cell_line_meta
