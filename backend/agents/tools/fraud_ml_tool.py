"""
FraudMLTool for machine learning-based fraud detection.

This tool provides ML-based fraud detection capabilities using Random Forest
and XGBoost models. It handles imbalanced datasets with SMOTE, performs
feature engineering, integrates with Neo4j for graph features, and provides
model explainability.
"""

import json
import logging
import os
import pickle
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from crewai_tools import BaseTool
from pydantic import BaseModel, Field

# Optional dependencies - will be imported only if available
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from backend.integrations.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

# Default model directory
DEFAULT_MODEL_DIR = Path("models")


class FraudMLInput(BaseModel):
    """Input model for fraud ML operations."""
    
    operation: str = Field(
        ...,
        description="Operation to perform: 'train', 'predict', 'evaluate', 'explain'"
    )
    data_source: str = Field(
        ...,
        description="Source of data: 'csv', 'neo4j', 'dataframe'"
    )
    data_path: Optional[str] = Field(
        None,
        description="Path to CSV file or Neo4j query for data"
    )
    data: Optional[Dict[str, Any]] = Field(
        None,
        description="Data for prediction (when data_source is 'dataframe')"
    )
    model_type: str = Field(
        "random_forest",
        description="Model type: 'random_forest', 'xgboost', 'ensemble'"
    )
    target_column: Optional[str] = Field(
        None,
        description="Name of target column for training"
    )
    feature_columns: Optional[List[str]] = Field(
        None,
        description="List of feature columns to use"
    )
    categorical_columns: Optional[List[str]] = Field(
        None,
        description="List of categorical columns for encoding"
    )
    numerical_columns: Optional[List[str]] = Field(
        None,
        description="List of numerical columns for scaling"
    )
    graph_features: bool = Field(
        False,
        description="Whether to include Neo4j graph features"
    )
    use_smote: bool = Field(
        True,
        description="Whether to use SMOTE for handling imbalanced data"
    )
    model_params: Optional[Dict[str, Any]] = Field(
        None,
        description="Parameters for the model"
    )
    model_path: Optional[str] = Field(
        None,
        description="Path to save or load model"
    )
    test_size: float = Field(
        0.3,
        description="Proportion of data to use for testing"
    )
    random_state: int = Field(
        42,
        description="Random state for reproducibility"
    )
    crypto_dataset: bool = Field(
        False,
        description="Whether the dataset is for cryptocurrency transactions"
    )
    entity_id_column: Optional[str] = Field(
        None,
        description="Column containing entity IDs for graph feature extraction"
    )


class FraudMLTool(BaseTool):
    """
    Tool for machine learning-based fraud detection.
    
    This tool provides capabilities for training, evaluating, and using
    machine learning models for fraud detection. It supports Random Forest
    and XGBoost models, handles imbalanced data with SMOTE, and can
    integrate with Neo4j for graph-based features.
    """
    
    name: str = "fraud_ml_tool"
    description: str = """
    Detect fraud using machine learning models.
    
    Use this tool when you need to:
    - Train ML models on transaction data for fraud detection
    - Predict fraud likelihood for new transactions
    - Evaluate model performance with metrics like precision, recall, and F1
    - Explain model predictions with feature importance
    - Integrate graph features from Neo4j with traditional ML
    
    The tool supports both traditional finance and cryptocurrency datasets,
    handles imbalanced data using SMOTE, and provides explainability features.
    
    Example usage:
    - Train a Random Forest model on transaction data
    - Predict fraud probability for a set of transactions
    - Evaluate model performance on test data
    - Explain which features contribute most to fraud predictions
    """
    args_schema: type[BaseModel] = FraudMLInput
    
    def __init__(
        self,
        neo4j_client: Optional[Neo4jClient] = None,
        model_dir: Optional[Path] = None
    ):
        """
        Initialize the FraudMLTool.
        
        Args:
            neo4j_client: Optional Neo4jClient for graph features
            model_dir: Directory to store trained models
        """
        super().__init__()
        
        # Check if required libraries are available
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available. Install with 'pip install scikit-learn'")
        
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available. Install with 'pip install xgboost'")
        
        if not IMBLEARN_AVAILABLE:
            logger.warning("imbalanced-learn not available. Install with 'pip install imbalanced-learn'")
        
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Install with 'pip install shap'")
        
        # Initialize attributes
        self.neo4j_client = neo4j_client
        self.model_dir = model_dir or DEFAULT_MODEL_DIR
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize model and preprocessing components
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.categorical_columns = None
        self.numerical_columns = None
        self.target_column = None
    
    async def _arun(
        self,
        operation: str,
        data_source: str,
        data_path: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        model_type: str = "random_forest",
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None,
        graph_features: bool = False,
        use_smote: bool = True,
        model_params: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
        test_size: float = 0.3,
        random_state: int = 42,
        crypto_dataset: bool = False,
        entity_id_column: Optional[str] = None
    ) -> str:
        """
        Run the FraudMLTool asynchronously.
        
        Args:
            operation: Operation to perform ('train', 'predict', 'evaluate', 'explain')
            data_source: Source of data ('csv', 'neo4j', 'dataframe')
            data_path: Path to CSV file or Neo4j query
            data: Data for prediction (when data_source is 'dataframe')
            model_type: Type of model to use
            target_column: Name of target column for training
            feature_columns: List of feature columns to use
            categorical_columns: List of categorical columns for encoding
            numerical_columns: List of numerical columns for scaling
            graph_features: Whether to include Neo4j graph features
            use_smote: Whether to use SMOTE for handling imbalanced data
            model_params: Parameters for the model
            model_path: Path to save or load model
            test_size: Proportion of data to use for testing
            random_state: Random state for reproducibility
            crypto_dataset: Whether the dataset is for cryptocurrency transactions
            entity_id_column: Column containing entity IDs for graph feature extraction
            
        Returns:
            JSON string with operation results
        """
        try:
            # Check if required libraries are available
            if not SKLEARN_AVAILABLE:
                return json.dumps({
                    "success": False,
                    "error": "scikit-learn is not available. Install with 'pip install scikit-learn'"
                })
            
            if model_type == "xgboost" and not XGBOOST_AVAILABLE:
                return json.dumps({
                    "success": False,
                    "error": "XGBoost is not available. Install with 'pip install xgboost'"
                })
            
            if use_smote and not IMBLEARN_AVAILABLE:
                return json.dumps({
                    "success": False,
                    "error": "imbalanced-learn is not available. Install with 'pip install imbalanced-learn'"
                })
            
            # Load data
            df = await self._load_data(data_source, data_path, data, crypto_dataset)
            if df is None:
                return json.dumps({
                    "success": False,
                    "error": "Failed to load data"
                })
            
            # Add graph features if requested
            if graph_features and entity_id_column:
                df = await self._add_graph_features(df, entity_id_column)
            
            # Perform requested operation
            if operation == "train":
                result = await self._train_model(
                    df,
                    model_type,
                    target_column,
                    feature_columns,
                    categorical_columns,
                    numerical_columns,
                    use_smote,
                    model_params,
                    model_path,
                    test_size,
                    random_state,
                    crypto_dataset
                )
            elif operation == "predict":
                result = await self._predict(
                    df,
                    model_path,
                    feature_columns
                )
            elif operation == "evaluate":
                result = await self._evaluate_model(
                    df,
                    model_path,
                    target_column,
                    feature_columns,
                    categorical_columns,
                    numerical_columns
                )
            elif operation == "explain":
                result = await self._explain_predictions(
                    df,
                    model_path,
                    feature_columns,
                    categorical_columns,
                    numerical_columns
                )
            else:
                return json.dumps({
                    "success": False,
                    "error": f"Unknown operation: {operation}"
                })
            
            return json.dumps(result)
            
        except Exception as e:
            logger.error(f"Error in FraudMLTool: {e}", exc_info=True)
            return json.dumps({
                "success": False,
                "error": str(e)
            })
    
    def _run(
        self,
        operation: str,
        data_source: str,
        data_path: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        model_type: str = "random_forest",
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None,
        graph_features: bool = False,
        use_smote: bool = True,
        model_params: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
        test_size: float = 0.3,
        random_state: int = 42,
        crypto_dataset: bool = False,
        entity_id_column: Optional[str] = None
    ) -> str:
        """
        Synchronous wrapper for _arun.
        
        This method exists for compatibility with synchronous CrewAI operations.
        It should not be called directly in an async context.
        """
        import asyncio
        
        # Create a new event loop if needed
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self._arun(
                operation,
                data_source,
                data_path,
                data,
                model_type,
                target_column,
                feature_columns,
                categorical_columns,
                numerical_columns,
                graph_features,
                use_smote,
                model_params,
                model_path,
                test_size,
                random_state,
                crypto_dataset,
                entity_id_column
            )
        )
    
    async def _load_data(
        self,
        data_source: str,
        data_path: Optional[str],
        data: Optional[Dict[str, Any]],
        crypto_dataset: bool
    ) -> Optional[pd.DataFrame]:
        """
        Load data from the specified source.
        
        Args:
            data_source: Source of data ('csv', 'neo4j', 'dataframe')
            data_path: Path to CSV file or Neo4j query
            data: Data for prediction (when data_source is 'dataframe')
            crypto_dataset: Whether the dataset is for cryptocurrency transactions
            
        Returns:
            Pandas DataFrame or None if loading fails
        """
        try:
            if data_source == "csv":
                if not data_path:
                    logger.error("data_path must be provided for CSV data source")
                    return None
                
                df = pd.read_csv(data_path)
                logger.info(f"Loaded {len(df)} rows from CSV file")
                
            elif data_source == "neo4j":
                if not data_path:
                    logger.error("data_path must be provided for Neo4j data source")
                    return None
                
                if not self.neo4j_client:
                    logger.error("Neo4j client not available")
                    return None
                
                # Execute Neo4j query
                results = await self.neo4j_client.run_query(data_path)
                
                # Convert to DataFrame
                df = pd.DataFrame(results)
                logger.info(f"Loaded {len(df)} rows from Neo4j")
                
            elif data_source == "dataframe":
                if not data:
                    logger.error("data must be provided for dataframe data source")
                    return None
                
                # Convert dict to DataFrame
                df = pd.DataFrame(data)
                logger.info(f"Loaded {len(df)} rows from provided data")
                
            else:
                logger.error(f"Unknown data source: {data_source}")
                return None
            
            # Perform dataset-specific preprocessing
            if crypto_dataset:
                df = self._preprocess_crypto_dataset(df)
            else:
                df = self._preprocess_traditional_dataset(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            return None
    
    def _preprocess_traditional_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess traditional finance dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        # Handle common preprocessing for traditional finance data
        
        # Convert timestamp columns to datetime
        for col in df.columns:
            if "time" in col.lower() or "date" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
        
        # Handle missing values in numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Add derived features for traditional finance
        df = self._add_traditional_finance_features(df)
        
        return df
    
    def _preprocess_crypto_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess cryptocurrency dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        # Handle common preprocessing for crypto data
        
        # Convert timestamp columns to datetime
        for col in df.columns:
            if "time" in col.lower() or "date" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
        
        # Handle wallet addresses (convert to categorical if needed)
        address_cols = [col for col in df.columns if "address" in col.lower() or "wallet" in col.lower()]
        for col in address_cols:
            if df[col].dtype == "object":
                # Create a hash feature from the address
                df[f"{col}_hash"] = df[col].apply(lambda x: hash(str(x)) % 10000)
        
        # Add derived features for crypto
        df = self._add_crypto_features(df)
        
        return df
    
    def _add_traditional_finance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features for traditional finance data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
        # Find amount and timestamp columns
        amount_cols = [col for col in df.columns if "amount" in col.lower() or "value" in col.lower()]
        time_cols = [col for col in df.columns if "time" in col.lower() or "date" in col.lower()]
        
        if amount_cols and time_cols:
            amount_col = amount_cols[0]
            time_col = time_cols[0]
            
            # Check if time column is datetime
            if pd.api.types.is_datetime64_dtype(df[time_col]):
                # Add hour of day feature
                df["hour_of_day"] = df[time_col].dt.hour
                
                # Add day of week feature
                df["day_of_week"] = df[time_col].dt.dayofweek
                
                # Add weekend indicator
                df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
                
                # Add month feature
                df["month"] = df[time_col].dt.month
            
            # Add amount-based features
            if pd.api.types.is_numeric_dtype(df[amount_col]):
                # Add rounded amount feature (for detecting just-below threshold transactions)
                df["amount_rounded"] = (df[amount_col] / 1000).round() * 1000
                
                # Add flag for amounts just below common thresholds
                df["just_below_10k"] = ((df[amount_col] > 9000) & (df[amount_col] < 10000)).astype(int)
                df["just_below_5k"] = ((df[amount_col] > 4500) & (df[amount_col] < 5000)).astype(int)
                
                # Add amount bin feature
                df["amount_bin"] = pd.cut(
                    df[amount_col],
                    bins=[0, 1000, 5000, 10000, 50000, float('inf')],
                    labels=[1, 2, 3, 4, 5]
                ).astype(int)
        
        return df
    
    def _add_crypto_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features for cryptocurrency data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
        # Find amount and timestamp columns
        amount_cols = [col for col in df.columns if "amount" in col.lower() or "value" in col.lower()]
        time_cols = [col for col in df.columns if "time" in col.lower() or "date" in col.lower()]
        address_cols = [col for col in df.columns if "address" in col.lower() or "wallet" in col.lower()]
        
        if amount_cols:
            amount_col = amount_cols[0]
            
            # Add amount-based features
            if pd.api.types.is_numeric_dtype(df[amount_col]):
                # Add log transform of amount (common in crypto)
                df["log_amount"] = np.log1p(df[amount_col])
                
                # Add rounded amount feature
                df["amount_rounded"] = (df[amount_col] / 0.1).round() * 0.1
        
        # Add time-based features
        if time_cols:
            time_col = time_cols[0]
            
            if pd.api.types.is_datetime64_dtype(df[time_col]):
                # Add hour of day feature
                df["hour_of_day"] = df[time_col].dt.hour
                
                # Add time zone related features (crypto is 24/7 global)
                df["utc_hour"] = df[time_col].dt.hour
                
                # Add day of week feature
                df["day_of_week"] = df[time_col].dt.dayofweek
        
        # Add address-based features
        if len(address_cols) >= 2:
            # Check for self-transactions (same from/to address)
            from_col = address_cols[0]
            to_col = address_cols[1]
            
            if from_col in df.columns and to_col in df.columns:
                df["is_self_transaction"] = (df[from_col] == df[to_col]).astype(int)
        
        return df
    
    async def _add_graph_features(
        self,
        df: pd.DataFrame,
        entity_id_column: str
    ) -> pd.DataFrame:
        """
        Add graph-based features from Neo4j.
        
        Args:
            df: Input DataFrame
            entity_id_column: Column containing entity IDs
            
        Returns:
            DataFrame with additional graph features
        """
        if not self.neo4j_client:
            logger.warning("Neo4j client not available, skipping graph features")
            return df
        
        if entity_id_column not in df.columns:
            logger.warning(f"Entity ID column '{entity_id_column}' not found in DataFrame")
            return df
        
        try:
            # Get unique entity IDs
            entity_ids = df[entity_id_column].unique().tolist()
            
            # Prepare Cypher query for PageRank scores
            pagerank_query = """
            CALL gds.pageRank.stream('financial_graph')
            YIELD nodeId, score
            MATCH (n) WHERE id(n) = nodeId AND n.id IN $entity_ids
            RETURN n.id AS entity_id, score AS pagerank_score
            """
            
            # Execute query
            pagerank_results = await self.neo4j_client.run_query(
                pagerank_query,
                {"entity_ids": entity_ids}
            )
            
            # Convert to DataFrame
            pagerank_df = pd.DataFrame(pagerank_results)
            
            if not pagerank_df.empty:
                # Merge with original DataFrame
                df = df.merge(
                    pagerank_df,
                    left_on=entity_id_column,
                    right_on="entity_id",
                    how="left"
                )
                
                # Fill missing values
                df["pagerank_score"] = df["pagerank_score"].fillna(0)
            
            # Add more graph features (community detection, centrality, etc.)
            # Community detection
            community_query = """
            CALL gds.louvain.stream('financial_graph')
            YIELD nodeId, communityId
            MATCH (n) WHERE id(n) = nodeId AND n.id IN $entity_ids
            RETURN n.id AS entity_id, communityId AS community_id
            """
            
            community_results = await self.neo4j_client.run_query(
                community_query,
                {"entity_ids": entity_ids}
            )
            
            community_df = pd.DataFrame(community_results)
            
            if not community_df.empty:
                # Merge with original DataFrame
                df = df.merge(
                    community_df,
                    left_on=entity_id_column,
                    right_on="entity_id",
                    how="left"
                )
                
                # Fill missing values and convert to category
                df["community_id"] = df["community_id"].fillna(-1).astype(int)
            
            # Betweenness centrality
            betweenness_query = """
            CALL gds.betweenness.stream('financial_graph')
            YIELD nodeId, score
            MATCH (n) WHERE id(n) = nodeId AND n.id IN $entity_ids
            RETURN n.id AS entity_id, score AS betweenness_score
            """
            
            betweenness_results = await self.neo4j_client.run_query(
                betweenness_query,
                {"entity_ids": entity_ids}
            )
            
            betweenness_df = pd.DataFrame(betweenness_results)
            
            if not betweenness_df.empty:
                # Merge with original DataFrame
                df = df.merge(
                    betweenness_df,
                    left_on=entity_id_column,
                    right_on="entity_id",
                    how="left"
                )
                
                # Fill missing values
                df["betweenness_score"] = df["betweenness_score"].fillna(0)
            
            # Transaction count (degree)
            degree_query = """
            MATCH (n)-[r]-(m)
            WHERE n.id IN $entity_ids
            RETURN n.id AS entity_id, count(r) AS transaction_count
            """
            
            degree_results = await self.neo4j_client.run_query(
                degree_query,
                {"entity_ids": entity_ids}
            )
            
            degree_df = pd.DataFrame(degree_results)
            
            if not degree_df.empty:
                # Merge with original DataFrame
                df = df.merge(
                    degree_df,
                    left_on=entity_id_column,
                    right_on="entity_id",
                    how="left"
                )
                
                # Fill missing values
                df["transaction_count"] = df["transaction_count"].fillna(0)
            
            logger.info(f"Added graph features for {len(entity_ids)} entities")
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding graph features: {e}", exc_info=True)
            return df
    
    async def _train_model(
        self,
        df: pd.DataFrame,
        model_type: str,
        target_column: str,
        feature_columns: Optional[List[str]],
        categorical_columns: Optional[List[str]],
        numerical_columns: Optional[List[str]],
        use_smote: bool,
        model_params: Optional[Dict[str, Any]],
        model_path: Optional[str],
        test_size: float,
        random_state: int,
        crypto_dataset: bool
    ) -> Dict[str, Any]:
        """
        Train a machine learning model for fraud detection.
        
        Args:
            df: Input DataFrame
            model_type: Type of model to use
            target_column: Name of target column
            feature_columns: List of feature columns
            categorical_columns: List of categorical columns
            numerical_columns: List of numerical columns
            use_smote: Whether to use SMOTE
            model_params: Parameters for the model
            model_path: Path to save model
            test_size: Proportion of data for testing
            random_state: Random state for reproducibility
            crypto_dataset: Whether using crypto dataset
            
        Returns:
            Dictionary with training results
        """
        try:
            if target_column not in df.columns:
                return {
                    "success": False,
                    "error": f"Target column '{target_column}' not found in DataFrame"
                }
            
            # Determine feature columns if not provided
            if not feature_columns:
                feature_columns = [col for col in df.columns if col != target_column]
            
            # Filter DataFrame to include only relevant columns
            columns_to_use = feature_columns + [target_column]
            df = df[columns_to_use].copy()
            
            # Handle missing values
            df = df.dropna(subset=[target_column])
            
            # Determine categorical and numerical columns if not provided
            if not categorical_columns:
                categorical_columns = df[feature_columns].select_dtypes(include=["object", "category"]).columns.tolist()
            
            if not numerical_columns:
                numerical_columns = df[feature_columns].select_dtypes(include=["number"]).columns.tolist()
            
            # Store column information
            self.feature_names = feature_columns
            self.categorical_columns = categorical_columns
            self.numerical_columns = numerical_columns
            self.target_column = target_column
            
            # Split data into features and target
            X = df[feature_columns]
            y = df[target_column]
            
            # Split into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Create preprocessing pipeline
            preprocessor = self._create_preprocessor(
                categorical_columns, numerical_columns
            )
            
            # Fit preprocessor on training data
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)
            
            # Store preprocessor
            self.preprocessor = preprocessor
            
            # Apply SMOTE for imbalanced data if requested
            if use_smote and IMBLEARN_AVAILABLE:
                # Check class distribution
                class_counts = np.bincount(y_train)
                if len(class_counts) > 1 and min(class_counts) > 0:
                    smote = SMOTE(random_state=random_state)
                    X_train_processed, y_train = smote.fit_resample(X_train_processed, y_train)
                    logger.info(f"Applied SMOTE: {np.bincount(y_train)}")
                else:
                    logger.warning("Skipping SMOTE due to insufficient class samples")
            
            # Create and train model
            model = self._create_model(model_type, model_params, random_state)
            
            # Train model with class weights for imbalanced data
            if model_type == "random_forest" or model_type == "ensemble":
                # Calculate class weights
                class_counts = np.bincount(y_train)
                if len(class_counts) > 1:
                    class_weight = {
                        0: 1.0,
                        1: class_counts[0] / class_counts[1] if class_counts[1] > 0 else 10.0
                    }
                    
                    # Train with class weights
                    model.fit(X_train_processed, y_train, sample_weight=None)
                else:
                    model.fit(X_train_processed, y_train)
            else:
                # XGBoost handles class weights differently
                if model_type == "xgboost" and XGBOOST_AVAILABLE:
                    class_counts = np.bincount(y_train)
                    if len(class_counts) > 1 and class_counts[1] > 0:
                        scale_pos_weight = class_counts[0] / class_counts[1]
                        model.set_params(scale_pos_weight=scale_pos_weight)
                
                model.fit(X_train_processed, y_train)
            
            # Store model
            self.model = model
            
            # Evaluate model on test set
            y_pred = model.predict(X_test_processed)
            
            # Get probabilities if available
            try:
                y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
            except:
                y_pred_proba = y_pred
            
            # Calculate metrics
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            try:
                auc = roc_auc_score(y_test, y_pred_proba)
            except:
                auc = 0.5
            
            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            
            # Save model if path provided
            if model_path:
                self._save_model(model_path)
                logger.info(f"Model saved to {model_path}")
            
            # Get feature importance
            feature_importance = self._get_feature_importance()
            
            # Return results
            return {
                "success": True,
                "metrics": {
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "auc": float(auc),
                    "true_positives": int(tp),
                    "false_positives": int(fp),
                    "true_negatives": int(tn),
                    "false_negatives": int(fn)
                },
                "model_type": model_type,
                "feature_importance": feature_importance,
                "class_distribution": {
                    "train": dict(zip(*np.unique(y_train, return_counts=True))),
                    "test": dict(zip(*np.unique(y_test, return_counts=True)))
                },
                "model_path": model_path
            }
            
        except Exception as e:
            logger.error(f"Error training model: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_preprocessor(
        self,
        categorical_columns: List[str],
        numerical_columns: List[str]
    ) -> ColumnTransformer:
        """
        Create a preprocessing pipeline for the data.
        
        Args:
            categorical_columns: List of categorical columns
            numerical_columns: List of numerical columns
            
        Returns:
            Scikit-learn ColumnTransformer
        """
        # Create transformers for different column types
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_columns),
                ('cat', categorical_transformer, categorical_columns)
            ],
            remainder='drop'
        )
        
        return preprocessor
    
    def _create_model(
        self,
        model_type: str,
        model_params: Optional[Dict[str, Any]],
        random_state: int
    ) -> Any:
        """
        Create a machine learning model.
        
        Args:
            model_type: Type of model to create
            model_params: Parameters for the model
            random_state: Random state for reproducibility
            
        Returns:
            Scikit-learn or XGBoost model
        """
        # Set default parameters if not provided
        if not model_params:
            model_params = {}
        
        # Create model based on type
        if model_type == "random_forest":
            default_params = {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": random_state,
                "class_weight": "balanced"
            }
            # Update with provided parameters
            default_params.update(model_params)
            
            return RandomForestClassifier(**default_params)
            
        elif model_type == "xgboost" and XGBOOST_AVAILABLE:
            default_params = {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": random_state,
                "objective": "binary:logistic"
            }
            # Update with provided parameters
            default_params.update(model_params)
            
            return xgb.XGBClassifier(**default_params)
            
        elif model_type == "ensemble":
            # Create both models
            rf_params = {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": random_state,
                "class_weight": "balanced"
            }
            
            if model_params.get("rf_params"):
                rf_params.update(model_params["rf_params"])
            
            rf_model = RandomForestClassifier(**rf_params)
            
            if XGBOOST_AVAILABLE:
                xgb_params = {
                    "n_estimators": 100,
                    "max_depth": 5,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": random_state,
                    "objective": "binary:logistic"
                }
                
                if model_params.get("xgb_params"):
                    xgb_params.update(model_params["xgb_params"])
                
                xgb_model = xgb.XGBClassifier(**xgb_params)
                
                # Use voting classifier for ensemble
                from sklearn.ensemble import VotingClassifier
                ensemble = VotingClassifier(
                    estimators=[
                        ('rf', rf_model),
                        ('xgb', xgb_model)
                    ],
                    voting='soft'
                )
                
                return ensemble
            else:
                # Fall back to Random Forest if XGBoost not available
                logger.warning("XGBoost not available, using Random Forest for ensemble")
                return rf_model
        else:
            # Default to Random Forest
            logger.warning(f"Unknown model type: {model_type}, using Random Forest")
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                class_weight="balanced"
            )
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None or self.preprocessor is None:
            return {}
        
        try:
            # Get feature names after preprocessing
            feature_names = []
            
            # Get numerical feature names (these stay the same)
            if self.numerical_columns:
                feature_names.extend(self.numerical_columns)
            
            # Get one-hot encoded feature names
            if self.categorical_columns:
                # Get the one-hot encoder
                try:
                    ohe = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
                    categories = ohe.categories_
                    
                    for i, col in enumerate(self.categorical_columns):
                        for cat in categories[i]:
                            feature_names.append(f"{col}_{cat}")
                except:
                    # Fallback if we can't get the categories
                    for col in self.categorical_columns:
                        feature_names.append(f"{col}_encoded")
            
            # Get feature importance from model
            importance = None
            
            if hasattr(self.model, "feature_importances_"):
                # Random Forest or XGBoost
                importance = self.model.feature_importances_
            elif hasattr(self.model, "coef_"):
                # Linear models
                importance = np.abs(self.model.coef_[0])
            elif hasattr(self.model, "estimators_"):
                # Voting Classifier or other ensemble
                if hasattr(self.model.estimators_[0], "feature_importances_"):
                    # Average feature importance across estimators
                    importance = np.mean([
                        est.feature_importances_ for est in self.model.estimators_
                        if hasattr(est, "feature_importances_")
                    ], axis=0)
            
            if importance is not None and len(importance) == len(feature_names):
                # Create dictionary mapping features to importance
                importance_dict = dict(zip(feature_names, importance))
                
                # Sort by importance
                importance_dict = {
                    k: float(v) for k, v in sorted(
                        importance_dict.items(),
                        key=lambda item: item[1],
                        reverse=True
                    )
                }
                
                return importance_dict
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}", exc_info=True)
            return {}
    
    async def _predict(
        self,
        df: pd.DataFrame,
        model_path: Optional[str],
        feature_columns: Optional[List[str]]
    ) -> Dict[str, Any]:
        """
        Make predictions using a trained model.
        
        Args:
            df: Input DataFrame
            model_path: Path to load model from
            feature_columns: List of feature columns
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Load model if path provided
            if model_path:
                self._load_model(model_path)
            
            # Check if model is loaded
            if self.model is None or self.preprocessor is None:
                return {
                    "success": False,
                    "error": "Model not loaded. Train a model first or provide model_path."
                }
            
            # Use stored feature columns if not provided
            if not feature_columns and self.feature_names:
                feature_columns = self.feature_names
            
            # Check if all required feature columns are present
            missing_columns = [col for col in feature_columns if col not in df.columns]
            if missing_columns:
                return {
                    "success": False,
                    "error": f"Missing columns in input data: {missing_columns}"
                }
            
            # Extract features
            X = df[feature_columns]
            
            # Apply preprocessing
            X_processed = self.preprocessor.transform(X)
            
            # Make predictions
            y_pred = self.model.predict(X_processed)
            
            # Get probabilities if available
            try:
                y_pred_proba = self.model.predict_proba(X_processed)[:, 1]
            except:
                y_pred_proba = y_pred
            
            # Add predictions to DataFrame
            df_with_predictions = df.copy()
            df_with_predictions["fraud_prediction"] = y_pred
            df_with_predictions["fraud_probability"] = y_pred_proba
            
            # Get feature importance for explainability
            feature_importance = self._get_feature_importance()
            
            # If SHAP is available, get SHAP values for explainability
            shap_values = None
            if SHAP_AVAILABLE and len(df) <= 1000:  # Limit SHAP to reasonable size
                try:
                    explainer = shap.Explainer(self.model, self.preprocessor.transform(df[feature_columns].iloc[:10]))
                    shap_values = explainer(self.preprocessor.transform(X))
                    
                    # Convert SHAP values to dictionary
                    shap_dict = {}
                    for i in range(min(10, len(df))):  # Limit to 10 examples
                        shap_dict[f"example_{i}"] = {
                            "base_value": float(explainer.expected_value) if hasattr(explainer, "expected_value") else 0,
                            "values": [float(v) for v in shap_values.values[i]],
                            "features": feature_columns
                        }
                except Exception as e:
                    logger.warning(f"Error calculating SHAP values: {e}")
                    shap_dict = {}
            else:
                shap_dict = {}
            
            # Return results
            return {
                "success": True,
                "predictions": {
                    "fraud_predictions": y_pred.tolist(),
                    "fraud_probabilities": y_pred_proba.tolist()
                },
                "prediction_counts": {
                    "fraud": int(np.sum(y_pred == 1)),
                    "non_fraud": int(np.sum(y_pred == 0))
                },
                "feature_importance": feature_importance,
                "shap_values": shap_dict
            }
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _evaluate_model(
        self,
        df: pd.DataFrame,
        model_path: Optional[str],
        target_column: str,
        feature_columns: Optional[List[str]],
        categorical_columns: Optional[List[str]],
        numerical_columns: Optional[List[str]]
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model on test data.
        
        Args:
            df: Input DataFrame
            model_path: Path to load model from
            target_column: Name of target column
            feature_columns: List of feature columns
            categorical_columns: List of categorical columns
            numerical_columns: List of numerical columns
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            # Check if target column exists
            if target_column not in df.columns:
                return {
                    "success": False,
                    "error": f"Target column '{target_column}' not found in DataFrame"
                }
            
            # Load model if path provided
            if model_path:
                self._load_model(model_path)
            
            # Check if model is loaded
            if self.model is None or self.preprocessor is None:
                return {
                    "success": False,
                    "error": "Model not loaded. Train a model first or provide model_path."
                }
            
            # Use stored feature columns if not provided
            if not feature_columns and self.feature_names:
                feature_columns = self.feature_names
            
            # Use stored categorical and numerical columns if not provided
            if not categorical_columns and self.categorical_columns:
                categorical_columns = self.categorical_columns
            
            if not numerical_columns and self.numerical_columns:
                numerical_columns = self.numerical_columns
            
            # Check if all required feature columns are present
            missing_columns = [col for col in feature_columns if col not in df.columns]
            if missing_columns:
                return {
                    "success": False,
                    "error": f"Missing columns in input data: {missing_columns}"
                }
            
            # Extract features and target
            X = df[feature_columns]
            y = df[target_column]
            
            # Apply preprocessing
            X_processed = self.preprocessor.transform(X)
            
            # Make predictions
            y_pred = self.model.predict(X_processed)
            
            # Get probabilities if available
            try:
                y_pred_proba = self.model.predict_proba(X_processed)[:, 1]
            except:
                y_pred_proba = y_pred
            
            # Calculate metrics
            precision = precision_score(y, y_pred, zero_division=0)
            recall = recall_score(y, y_pred, zero_division=0)
            f1 = f1_score(y, y_pred, zero_division=0)
            
            try:
                auc = roc_auc_score(y, y_pred_proba)
            except:
                auc = 0.5
            
            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
            
            # Get feature importance
            feature_importance = self._get_feature_importance()
            
            # Return results
            return {
                "success": True,
                "metrics": {
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "auc": float(auc),
                    "true_positives": int(tp),
                    "false_positives": int(fp),
                    "true_negatives": int(tn),
                    "false_negatives": int(fn)
                },
                "feature_importance": feature_importance,
                "class_distribution": dict(zip(*np.unique(y, return_counts=True)))
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _explain_predictions(
        self,
        df: pd.DataFrame,
        model_path: Optional[str],
        feature_columns: Optional[List[str]],
        categorical_columns: Optional[List[str]],
        numerical_columns: Optional[List[str]]
    ) -> Dict[str, Any]:
        """
        Explain model predictions using feature importance and SHAP values.
        
        Args:
            df: Input DataFrame
            model_path: Path to load model from
            feature_columns: List of feature columns
            categorical_columns: List of categorical columns
            numerical_columns: List of numerical columns
            
        Returns:
            Dictionary with explanation results
        """
        try:
            # Load model if path provided
            if model_path:
                self._load_model(model_path)
            
            # Check if model is loaded
            if self.model is None or self.preprocessor is None:
                return {
                    "success": False,
                    "error": "Model not loaded. Train a model first or provide model_path."
                }
            
            # Use stored feature columns if not provided
            if not feature_columns and self.feature_names:
                feature_columns = self.feature_names
            
            # Use stored categorical and numerical columns if not provided
            if not categorical_columns and self.categorical_columns:
                categorical_columns = self.categorical_columns
            
            if not numerical_columns and self.numerical_columns:
                numerical_columns = self.numerical_columns
            
            # Check if all required feature columns are present
            missing_columns = [col for col in feature_columns if col not in df.columns]
            if missing_columns:
                return {
                    "success": False,
                    "error": f"Missing columns in input data: {missing_columns}"
                }
            
            # Extract features
            X = df[feature_columns]
            
            # Apply preprocessing
            X_processed = self.preprocessor.transform(X)
            
            # Make predictions
            y_pred = self.model.predict(X_processed)
            
            # Get probabilities if available
            try:
                y_pred_proba = self.model.predict_proba(X_processed)[:, 1]
            except:
                y_pred_proba = y_pred
            
            # Get feature importance
            feature_importance = self._get_feature_importance()
            
            # Get SHAP values for explainability
            shap_values = None
            if SHAP_AVAILABLE and len(df) <= 1000:  # Limit SHAP to reasonable size
                try:
                    explainer = shap.Explainer(self.model, self.preprocessor.transform(df[feature_columns].iloc[:10]))
                    shap_values = explainer(self.preprocessor.transform(X))
                    
                    # Convert SHAP values to dictionary
                    shap_dict = {}
                    for i in range(min(10, len(df))):  # Limit to 10 examples
                        shap_dict[f"example_{i}"] = {
                            "base_value": float(explainer.expected_value) if hasattr(explainer, "expected_value") else 0,
                            "values": [float(v) for v in shap_values.values[i]],
                            "features": feature_columns
                        }
                except Exception as e:
                    logger.warning(f"Error calculating SHAP values: {e}")
                    shap_dict = {}
            else:
                shap_dict = {}
            
            # Generate explanations for high-risk predictions
            explanations = []
            high_risk_indices = np.where(y_pred_proba > 0.7)[0][:5]  # Top 5 high-risk predictions
            
            for idx in high_risk_indices:
                explanation = {
                    "index": int(idx),
                    "probability": float(y_pred_proba[idx]),
                    "top_features": {}
                }
                
                # Add original feature values
                for col in feature_columns:
                    if col in df.columns:
                        value = df.iloc[idx][col]
                        if isinstance(value, (int, float, bool)):
                            explanation[col] = float(value)
                        else:
                            explanation[col] = str(value)
                
                # Add top contributing features based on feature importance
                if feature_importance:
                    # Get top 5 important features
                    top_features = list(feature_importance.keys())[:5]
                    for feature in top_features:
                        if feature in df.columns:
                            value = df.iloc[idx][feature]
                            if isinstance(value, (int, float, bool)):
                                explanation["top_features"][feature] = {
                                    "value": float(value),
                                    "importance": feature_importance.get(feature, 0)
                                }
                            else:
                                explanation["top_features"][feature] = {
                                    "value": str(value),
                                    "importance": feature_importance.get(feature, 0)
                                }
                
                explanations.append(explanation)
            
            # Return results
            return {
                "success": True,
                "feature_importance": feature_importance,
                "shap_values": shap_dict,
                "high_risk_explanations": explanations,
                "model_type": type(self.model).__name__
            }
            
        except Exception as e:
            logger.error(f"Error explaining predictions: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def _save_model(self, model_path: str) -> bool:
        """
        Save the trained model and preprocessor.
        
        Args:
            model_path: Path to save model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
            
            # Save model, preprocessor, and metadata
            model_data = {
                "model": self.model,
                "preprocessor": self.preprocessor,
                "feature_names": self.feature_names,
                "categorical_columns": self.categorical_columns,
                "numerical_columns": self.numerical_columns,
                "target_column": self.target_column,
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "model_type": type(self.model).__name__
                }
            }
            
            with open(model_path, "wb") as f:
                pickle.dump(model_data, f)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}", exc_info=True)
            return False
    
    def _load_model(self, model_path: str) -> bool:
        """
        Load a trained model and preprocessor.
        
        Args:
            model_path: Path to load model from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Load model data
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)
            
            # Extract components
            self.model = model_data["model"]
            self.preprocessor = model_data["preprocessor"]
            self.feature_names = model_data["feature_names"]
            self.categorical_columns = model_data["categorical_columns"]
            self.numerical_columns = model_data["numerical_columns"]
            self.target_column = model_data["target_column"]
            
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            return False
