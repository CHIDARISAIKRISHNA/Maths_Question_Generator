import json
import os
from typing import List, Dict, Any, Optional
import pandas as pd

class JsonlDataLoader:
    """
    Loads and processes data from JSONL files.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the JSONL data loader.
        
        Args:
            data_dir: Directory containing the JSONL files
        """
        self.data_dir = data_dir
        self.data = {}
    
    def load_jsonl_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load data from a JSONL file.
        
        Args:
            file_path: Path to the JSONL file
            
        Returns:
            List of dictionaries containing the data
        """
        data = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
            
            print(f"Successfully loaded {len(data)} records from {file_path}")
            return data
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return []
        except Exception as e:
            print(f"Error loading JSONL data: {e}")
            return []
    
    def load_all_datasets(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load all JSONL datasets in the data directory.
        
        Returns:
            Dictionary mapping dataset names to data
        """
        datasets = {}
        
        # List of dataset files to load
        dataset_files = [
            "example_model_solutions.jsonl",
            "test.jsonl",
            "test_socratic.jsonl",
            "train.jsonl",
            "train_socratic.jsonl"
        ]
        
        for file_name in dataset_files:
            file_path = os.path.join(self.data_dir, file_name)
            if os.path.exists(file_path):
                dataset_name = os.path.splitext(file_name)[0]
                datasets[dataset_name] = self.load_jsonl_file(file_path)
            else:
                print(f"Warning: Dataset file {file_path} not found")
        
        self.data = datasets
        return datasets
    
    def get_problems_and_solutions(self, dataset_name: str) -> List[Dict[str, str]]:
        """
        Extract problems and solutions from a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            List of dictionaries containing problems and solutions
        """
        if dataset_name not in self.data:
            print(f"Dataset {dataset_name} not loaded. Loading now...")
            file_path = os.path.join(self.data_dir, f"{dataset_name}.jsonl")
            self.data[dataset_name] = self.load_jsonl_file(file_path)
        
        problems_and_solutions = []
        
        for item in self.data[dataset_name]:
            problem = item.get('problem', '')
            solution = item.get('solution', '')
            
            problems_and_solutions.append({
                'problem': problem,
                'solution': solution
            })
        
        return problems_and_solutions
    
    def convert_to_dataframe(self, dataset_name: str) -> pd.DataFrame:
        """
        Convert a dataset to a pandas DataFrame.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            DataFrame containing the dataset
        """
        if dataset_name not in self.data:
            print(f"Dataset {dataset_name} not loaded. Loading now...")
            file_path = os.path.join(self.data_dir, f"{dataset_name}.jsonl")
            self.data[dataset_name] = self.load_jsonl_file(file_path)
        
        return pd.DataFrame(self.data[dataset_name])
    
    def get_sample_data(self, dataset_name: str, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get a sample of data from a dataset.
        
        Args:
            dataset_name: Name of the dataset
            n: Number of samples to get
            
        Returns:
            List of sample data
        """
        if dataset_name not in self.data:
            print(f"Dataset {dataset_name} not loaded. Loading now...")
            file_path = os.path.join(self.data_dir, f"{dataset_name}.jsonl")
            self.data[dataset_name] = self.load_jsonl_file(file_path)
        
        import random
        return random.sample(self.data[dataset_name], min(n, len(self.data[dataset_name])))

# Example usage
if __name__ == "__main__":
    loader = JsonlDataLoader()
    datasets = loader.load_all_datasets()
    
    for name, data in datasets.items():
        print(f"Dataset: {name}, Records: {len(data)}")
    
    # Get problems and solutions from the test dataset
    if 'test' in datasets:
        problems = loader.get_problems_and_solutions('test')
        print(f"Extracted {len(problems)} problems with solutions from test dataset")
        
        # Show a sample
        sample = loader.get_sample_data('test', 1)
        for i, item in enumerate(sample, 1):
            print(f"\nSample {i}:")
            print(f"Problem: {item.get('problem', '')[:100]}...")
            print(f"Solution: {item.get('solution', '')[:100]}...")