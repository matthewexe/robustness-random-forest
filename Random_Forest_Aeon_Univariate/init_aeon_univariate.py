#!/usr/bin/env python3

import argparse
import json
import datetime
import os
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from aeon.datasets import load_classification

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

from eu import store_monotonic_dict
from forest import store_forest
from sample import sklearn_sample_to_dict, store_sample
from skforest_to_forest import sklearn_forest_to_forest

# Configure logging - will be set up properly in main() based on arguments
logger = logging.getLogger(__name__)

# List of popular aeon univariate datasets
AVAILABLE_DATASETS = [
    # Small datasets (good for testing)
    'Coffee', 'ECG200', 'GunPoint', 'ItalyPowerDemand', 'Lightning2', 'Lightning7',
    'MedicalImages', 'MoteStrain', 'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2',
    'Symbols', 'SyntheticControl', 'TwoLeadECG', 'Wafer', 'Wine', 'Yoga',

    # Medium datasets
    'Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'CBF',
    'ChlorineConcentration', 'CinCECGTorso', 'Computers', 'CricketX', 'CricketY', 'CricketZ',
    'DiatomSizeReduction', 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect',
    'DistalPhalanxTW', 'Earthquakes', 'ECG5000', 'ECGFiveDays', 'ElectricDevices',
    'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords', 'Fish', 'FordA', 'FordB',

    # Large datasets (use with caution)
    'HandOutlines', 'Haptics', 'Herring', 'InlineSkate', 'InsectWingbeatSound',
    'LargeKitchenAppliances', 'Mallat', 'Meat', 'MelbournePedestrian', 'MiddlePhalanxOutlineAgeGroup',
    'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW', 'MixedShapesRegularTrain', 'MixedShapesSmallTrain',
    'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2', 'OliveOil', 'OSULeaf',
    'PhalangesOutlinesCorrect', 'Phoneme', 'Plane', 'ProximalPhalanxOutlineAgeGroup',
    'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'RefrigerationDevices',
    'ScreenType', 'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'StarlightCurves',
    'Strawberry', 'SwedishLeaf', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace',
    'TwoPatterns', 'UMD', 'UWaveGestureLibraryAll', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY',
    'UWaveGestureLibraryZ', 'WordSynonyms', 'Worms', 'WormsTwoClass'
]

RESULTS_DIR = "results"


def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Path to log file (default: None, uses results/init_aeon_univariate.log)
    """
    if log_file is None:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        log_file = os.path.join(RESULTS_DIR, 'init_aeon_univariate.log')
    else:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger.info(f"Logging configured: level={logging.getLevelName(log_level)}, file={log_file}")


def results_path(filename: str) -> str:
    """Return a path inside the results directory, creating it if needed."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, os.path.basename(filename))


def convert_numpy_types(obj):
    """
    Convert numpy types to native Python types for JSON serialization

    Args:
        obj: Object to convert (can be dict, list, or scalar)

    Returns:
        Converted object with native Python types
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def list_available_datasets():
    """Print available aeon univariate datasets"""
    print("Available Aeon Univariate Time Series Datasets:")
    print("=" * 50)

    print("\n Small Datasets (good for testing):")
    small_datasets = AVAILABLE_DATASETS[:16]
    for i, dataset in enumerate(small_datasets):
        if i % 4 == 0:
            print()
        print(f"  {dataset:<20}", end="")

    print(f"\n\n Medium Datasets:")
    medium_datasets = AVAILABLE_DATASETS[16:50]
    for i, dataset in enumerate(medium_datasets):
        if i % 3 == 0:
            print()
        print(f"  {dataset:<25}", end="")

    print(f"\n\n Large Datasets (use with caution):")
    large_datasets = AVAILABLE_DATASETS[50:]
    for i, dataset in enumerate(large_datasets):
        if i % 3 == 0:
            print()
        print(f"  {dataset:<25}", end="")

    print(f"\n\nTotal: {len(AVAILABLE_DATASETS)} datasets available")
    print("\nNote: Not all datasets may be available in your aeon installation.")


def get_dataset_info(dataset_name):
    """Try to get basic info about a dataset without fully loading it"""
    try:
        X_train, y_train = load_classification(dataset_name, split="train")
        X_test, y_test = load_classification(dataset_name, split="test")

        classes = np.unique(np.concatenate([y_train, y_test]))

        return {
            'train_size': X_train.shape[0],
            'test_size': X_test.shape[0],
            'series_length': X_train.shape[2],
            'n_channels': X_train.shape[1],
            'classes': classes.tolist(),
            'n_classes': len(classes)
        }
    except Exception as e:
        return {'error': str(e)}


def load_and_prepare_dataset(dataset_name, feature_prefix="t"):
    """Load and prepare aeon dataset"""
    logger.info(f"Loading dataset: {dataset_name}")
    print(f"Loading dataset: {dataset_name}")

    try:
        X_train, y_train = load_classification(dataset_name, split="train")
        X_test, y_test = load_classification(dataset_name, split="test")
        logger.info(f"Successfully loaded dataset from aeon")
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        raise Exception(f"Failed to load dataset {dataset_name}: {e}")

    print(f"Dataset: {dataset_name}")
    print(f"Training set: {X_train.shape} samples")
    print(f"Test set: {X_test.shape} samples")
    print(f"Series length: {X_train.shape[2]} time points")
    print(f"Classes: {np.unique(np.concatenate([y_train, y_test]))}")
    
    logger.debug(f"Dataset: {dataset_name}, Training set: {X_train.shape}, Test set: {X_test.shape}")
    logger.debug(f"Series length: {X_train.shape[2]}, Classes: {np.unique(np.concatenate([y_train, y_test]))}")

    # Reshape data: (n_samples, n_channels, n_timepoints) -> (n_samples, n_timepoints)
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    X_test_2d = X_test.reshape(X_test.shape[0], -1)
    logger.debug(f"Reshaped data: X_train_2d shape={X_train_2d.shape}, X_test_2d shape={X_test_2d.shape}")

    # Calculate padding width for feature names
    series_length = X_train_2d.shape[1]
    padding_width = len(str(series_length - 1))
    feature_names = [f"{feature_prefix}_{i:0{padding_width}d}" for i in range(series_length)]
    logger.debug(f"Generated {len(feature_names)} feature names with prefix '{feature_prefix}'")

    # Get all unique classes
    all_classes = np.unique(np.concatenate([y_train, y_test])).astype(str)
    logger.info(f"Dataset prepared with {len(all_classes)} classes: {all_classes}")

    return X_train_2d, y_train, X_test_2d, y_test, feature_names, all_classes


def create_forest_params(args):
    """Create RandomForest parameters from command line arguments"""
    rf_params = {'random_state': args.random_state}

    # Basic parameters
    if args.n_estimators:
        rf_params['n_estimators'] = args.n_estimators
    if args.criterion:
        rf_params['criterion'] = args.criterion

    # Tree structure parameters
    if args.max_depth:
        rf_params['max_depth'] = args.max_depth
    if args.min_samples_split:
        rf_params['min_samples_split'] = args.min_samples_split
    if args.min_samples_leaf:
        rf_params['min_samples_leaf'] = args.min_samples_leaf
    if args.max_leaf_nodes:
        rf_params['max_leaf_nodes'] = args.max_leaf_nodes

    # Feature selection
    if args.max_features:
        rf_params['max_features'] = args.max_features

    # Split quality
    if args.min_impurity_decrease:
        rf_params['min_impurity_decrease'] = args.min_impurity_decrease

    # Sampling parameters
    if args.bootstrap:
        rf_params['bootstrap'] = (args.bootstrap == 'True')
    if args.max_samples:
        rf_params['max_samples'] = args.max_samples

    # Pruning
    if args.ccp_alpha:
        rf_params['ccp_alpha'] = args.ccp_alpha

    return rf_params


def get_rf_search_space(include_bootstrap=True):
    """
    Define the hyperparameter search space for Bayesian optimization

    Args:
        include_bootstrap: If True, includes bootstrap and max_samples in search space.
                          If False, excludes them to avoid constraint violations (default: True)
    """

    search_space = {
        # Number of trees
        'n_estimators': Integer(10, 300, name='n_estimators'),

        # Tree structure
        'max_depth': Integer(2, 50, name='max_depth'),
        'min_samples_split': Integer(2, 20, name='min_samples_split'),
        'min_samples_leaf': Integer(1, 10, name='min_samples_leaf'),
        'max_leaf_nodes': Categorical([None, 10, 20, 30, 50, 100], name='max_leaf_nodes'),

        # Feature selection
        'max_features': Categorical(['sqrt', 'log2', None], name='max_features'),

        # Split quality
        'criterion': Categorical(['gini', 'entropy'], name='criterion'),
        'min_impurity_decrease': Real(0.0, 0.1, prior='uniform', name='min_impurity_decrease'),

        # Pruning
        'ccp_alpha': Real(0.0, 0.05, prior='uniform', name='ccp_alpha'),
    }

    # Only include bootstrap-related parameters if requested
    # Note: max_samples only works when bootstrap=True, so we keep bootstrap=True only
    if include_bootstrap:
        search_space['bootstrap'] = Categorical([True], name='bootstrap')  # Only True to avoid constraint
        search_space['max_samples'] = Categorical([None, 0.5, 0.7, 0.9], name='max_samples')

    return search_space


def optimize_rf_hyperparameters(X_train, y_train, search_space, n_iter=50, cv=5,
                                n_jobs=-1, random_state=42, verbose=1,
                                X_test=None, y_test=None, use_test_for_validation=False):
    """
    Perform Bayesian optimization to find best Random Forest hyperparameters

    Args:
        X_train: Training features
        y_train: Training labels
        search_space: Dictionary defining the search space for hyperparameters
        n_iter: Number of iterations for optimization (default: 50)
        cv: Number of cross-validation folds (default: 5)
        n_jobs: Number of parallel jobs (-1 for all cores)
        random_state: Random seed for reproducibility
        verbose: Verbosity level
        X_test: Test features (optional, for validation)
        y_test: Test labels (optional, for validation)
        use_test_for_validation: If True, uses test set for validation instead of CV (default: False)

    Returns:
        best_params: Dictionary of best hyperparameters found
        best_score: Best cross-validation/test score achieved
        test_score: Test set score (if test data provided)
        optimizer: The fitted BayesSearchCV object (if CV used) or best RF model
    """

    print(f" Starting Bayesian Optimization for Random Forest hyperparameters")
    print(f"   Search space: {len(search_space)} hyperparameters")
    print(f"   Iterations: {n_iter}")

    if use_test_for_validation:
        if X_test is None or y_test is None:
            raise ValueError("X_test and y_test must be provided when use_test_for_validation=True")
        print(f"   Validation: Test set ({X_test.shape[0]} samples)")
        print(f"     WARNING: Using test set for validation may lead to overfitting on test data!")
    else:
        print(f"   Validation: {cv}-fold cross-validation")

    print(f"   This may take a while...")

    if use_test_for_validation:
        # Manual optimization using test set for validation
        from skopt import gp_minimize
        from skopt.utils import use_named_args

        # Convert search space to list format for gp_minimize
        dimensions = list(search_space.values())

        best_score = -np.inf
        best_params = None
        best_model = None

        @use_named_args(dimensions)
        def objective(**params):
            nonlocal best_score, best_params, best_model

            # Handle constraint: max_samples can only be used with bootstrap=True
            if 'bootstrap' in params and 'max_samples' in params:
                if not params['bootstrap'] and params['max_samples'] is not None:
                    params['max_samples'] = None

            # Train model with current parameters
            rf = RandomForestClassifier(**params, random_state=random_state)
            rf.fit(X_train, y_train)

            # Evaluate on test set
            score = rf.score(X_test, y_test)

            # Track best model
            if score > best_score:
                best_score = score
                best_params = params.copy()
                best_model = rf

            # Return negative score (gp_minimize minimizes)
            return -score

        print("\n Running Bayesian optimization with test set validation...")
        result = gp_minimize(
            objective,
            dimensions,
            n_calls=n_iter,
            random_state=random_state,
            verbose=verbose > 0
        )

        print(f"\n Optimization complete!")
        print(f"   Best test score: {best_score:.4f}")
        print(f"   Best parameters:")
        for param, value in best_params.items():
            print(f"      {param}: {value}")

        return best_params, best_score, best_score, best_model

    else:
        # Standard cross-validation approach
        # Create base estimator
        rf = RandomForestClassifier(random_state=random_state)

        # BayesSearchCV doesn't automatically handle the bootstrap/max_samples constraint
        # We need to use a custom scorer or handle it differently
        # For now, we'll rely on sklearn's internal validation during fit

        # Create Bayesian optimizer
        optimizer = BayesSearchCV(
            estimator=rf,
            search_spaces=search_space,
            n_iter=n_iter,
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            scoring='accuracy',
            return_train_score=True,
            error_score='raise'  # Raise errors instead of ignoring them
        )

        # Fit the optimizer
        print("\n Running Bayesian optimization...")
        optimizer.fit(X_train, y_train)

        best_params = optimizer.best_params_
        best_score = optimizer.best_score_

        # Optionally evaluate on test set
        test_score = None
        if X_test is not None and y_test is not None:
            test_score = optimizer.best_estimator_.score(X_test, y_test)
            print(f"\n Optimization complete!")
            print(f"   Best CV score: {best_score:.4f}")
            print(f"   Test set score: {test_score:.4f}")
            print(f"   Best parameters:")
            for param, value in best_params.items():
                print(f"      {param}: {value}")
        else:
            print(f"\n Optimization complete!")
            print(f"   Best CV score: {best_score:.4f}")
            print(f"   Best parameters:")
            for param, value in best_params.items():
                print(f"      {param}: {value}")

        return best_params, best_score, test_score, optimizer


def train_and_convert_forest(X_train, y_train, X_test, y_test, rf_params, feature_names,
                             test_split=None, random_state=42, sample_percentage=None):
    """
    Train Random Forest and convert to our format

    Args:
        X_train: Training features (from aeon)
        y_train: Training labels (from aeon)
        X_test: Test features (from aeon)
        y_test: Test labels (from aeon)
        rf_params: Random Forest parameters
        feature_names: List of feature names
        test_split: If provided (e.g., 0.3), combines train+test and does custom split.
                   If None, uses the original aeon train/test split (default behavior)
        random_state: Random seed for reproducibility
        sample_percentage: If provided (e.g., 0.05), uses only a percentage of the combined data

    Returns:
        sklearn_rf: Trained sklearn RandomForestClassifier
        our_forest: Converted Forest object
        final_X_train: The actual training data used (for saving to DB)
        final_y_train: The actual training labels used (for saving to DB)
    """
    logger.info("Training Random Forest model")
    print("Training Random Forest...")

    if test_split is not None:
        # Custom split mode: combine train+test and re-split
        logger.info(f"Using custom split mode with test_size={test_split}")
        print(f"Custom split mode: combining datasets and splitting with test_size={test_split}")
        X_combined = np.vstack([X_train, X_test])
        y_combined = np.concatenate([y_train, y_test])

        # Apply sample percentage filtering if specified
        if sample_percentage is not None and sample_percentage < 100.0:
            logger.info(f"Applying sample percentage filter: {sample_percentage}%")
            print(f"Applying sample percentage filter: {sample_percentage}%")
            # Calculate how many samples to keep
            n_samples = len(X_combined)
            n_keep = int(n_samples * sample_percentage / 100.0)

            # Randomly select indices
            indices = np.random.choice(n_samples, size=n_keep, replace=False)

            # Filter the data
            X_combined = X_combined[indices]
            y_combined = y_combined[indices]

            print(f"Reduced from {n_samples} to {len(X_combined)} samples ({sample_percentage}%)")
            logger.debug(f"Reduced from {n_samples} to {len(X_combined)} samples")

        X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
            X_combined, y_combined, test_size=test_split, random_state=random_state
        )
        logger.debug(f"Split data: train={X_train_final.shape}, test={X_test_final.shape}")
    else:
        # Default mode: use original aeon train/test split
        logger.info("Using original aeon train/test split")
        print("Using original aeon train/test split")
        X_train_final = X_train
        y_train_final = y_train
        X_test_final = X_test
        y_test_final = y_test

    print(f"Training set: {X_train_final.shape[0]} samples")
    print(f"Test set: {X_test_final.shape[0]} samples")
    logger.info(f"Training set: {X_train_final.shape[0]} samples, Test set: {X_test_final.shape[0]} samples")

    # Train Random Forest
    logger.debug(f"Training Random Forest with parameters: {rf_params}")
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_train_final, y_train_final)
    logger.info("Random Forest training completed")

    # Evaluate
    train_score = rf.score(X_train_final, y_train_final)
    test_score = rf.score(X_test_final, y_test_final)

    print(f"Training accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")
    logger.info(f"Model evaluation: training_accuracy={train_score:.3f}, test_accuracy={test_score:.3f}")

    # Convert to our Forest format
    logger.info("Converting sklearn Random Forest to custom Forest format")
    print("Converting to custom Forest format...")
    our_forest = sklearn_forest_to_forest(rf, feature_names)
    print(f"Converted to Forest with {len(our_forest)} trees")
    logger.info(f"Conversion complete: {len(our_forest)} trees in custom format")

    return rf, our_forest, X_train_final, y_train_final


def store_training_set(X_train, y_train, feature_names, dataset_name):
    training_data = {
        'X_train': X_train.tolist(),  # Convert numpy array to list for JSON serialization
        'y_train': y_train.tolist(),
        'feature_names': feature_names,
        'dataset_name': dataset_name,
        'n_samples': X_train.shape[0],
        'n_features': X_train.shape[1],
        'timestamp': datetime.datetime.now().isoformat()
    }

    try:
        filename = results_path(f"{dataset_name}_training_set.json")
        with open(filename, 'w') as f:
            f.write(json.dumps(training_data))

        print(f"Training set saved successfully ({X_train.shape[0]} samples, {X_train.shape[1]} features)")
        return True
    except Exception as e:
        print(f"Failed to save training set: {e}")
        return False


def process_all_classified_samples(dataset_name, class_label, our_forest,
                                   X_test, y_test, feature_names, sample_percentage=None):
    """
    Process all test samples that are classified with the specified class label
    Store samples in DATA and their ICF representations in R

    If sample_percentage is provided, only process that percentage of samples
    """
    logger.info(f"Processing all samples classified as '{class_label}'")
    print(f"\n=== Processing All Samples Classified as '{class_label}' ===")

    # Find all test samples that are classified as the target class
    target_samples_data = []
    current_time = datetime.datetime.now().isoformat()

    # Apply sample percentage filtering if specified
    total_test_samples = len(X_test)
    if sample_percentage is not None and sample_percentage < 100.0:
        logger.info(f"Applying sample percentage filter: {sample_percentage}% of {total_test_samples} test samples")
        print(f"Applying sample percentage filter: {sample_percentage}% of {total_test_samples} test samples")
        # Randomly select indices
        n_keep = int(total_test_samples * sample_percentage / 100.0)
        indices = np.random.choice(total_test_samples, size=n_keep, replace=False)
        # Filter X_test and y_test
        X_test_filtered = X_test[indices]
        y_test_filtered = y_test[indices]
        print(f"Reduced test samples from {total_test_samples} to {len(X_test_filtered)} ({sample_percentage}%)")
        logger.debug(f"Reduced test samples from {total_test_samples} to {len(X_test_filtered)}")
    else:
        X_test_filtered = X_test
        y_test_filtered = y_test

    logger.debug(f"Processing {len(X_test_filtered)} test samples for classification")
    for i, (sample, actual_label) in enumerate(zip(X_test_filtered, y_test_filtered)):
        sample_dict = sklearn_sample_to_dict(sample, feature_names)
        predicted_label = our_forest.predict(sample_dict)

        # Store ALL samples classified with the target label (regardless of correctness)
        if predicted_label == class_label:
            target_samples_data.append({
                'test_index': i,
                'sample_dict': sample_dict,
                'predicted_label': predicted_label,
                'actual_label': actual_label,
                'prediction_correct': (predicted_label == actual_label)
            })

    print(f"Found {len(target_samples_data)} samples classified as '{class_label}'")
    logger.info(f"Found {len(target_samples_data)} samples classified as '{class_label}'")

    if len(target_samples_data) == 0:
        print("  No samples classified with the target label!")
        logger.warning(f"No samples classified with target label '{class_label}'")
        return [], {}

    # Store all samples and their ICF representations
    stored_samples = []
    correct_predictions = 0

    logger.debug(f"Storing {len(target_samples_data)} samples")
    for idx, sample_data in enumerate(target_samples_data):
        filename = f"{dataset_name}_{class_label}_{idx}.json"

        # Store sample in DATA with full metadata
        data_entry = {
            'sample_dict': sample_data['sample_dict'],
            'predicted_label': sample_data['predicted_label'],
            'actual_label': sample_data['actual_label'],
            'test_index': sample_data['test_index'],
            'dataset_name': dataset_name,
            'timestamp': current_time,
            'prediction_correct': sample_data['prediction_correct']
        }

        if store_sample(sample_data['sample_dict'], "sample_" + filename):
            if sample_data['prediction_correct']:
                correct_predictions += 1
            stored_samples.append({
                'sample_key': "sample_" + filename,
                **data_entry
            })
            # Also store full metadata separately
            meta_path = results_path("sample_meta_" + filename)
            with open(meta_path, 'w') as f:
                f.write(json.dumps(data_entry))


    # Store summary information
    summary = {
        'dataset_name': dataset_name,
        'target_class_label': class_label,
        'total_samples_processed': len(stored_samples),
        'total_test_samples': len(X_test_filtered),
        'samples_with_target_label': len(target_samples_data),
        'correct_predictions': correct_predictions,
        'incorrect_predictions': len(stored_samples) - correct_predictions,
        'target_label_precision': correct_predictions / len(stored_samples) if len(stored_samples) > 0 else 0.0,
        'timestamp': current_time,
        'sample_keys': [s['sample_key'] for s in stored_samples]
    }

    filename = f"summary_{dataset_name}_{class_label}.json"
    summary_path = results_path("sample_meta_" + filename)
    with open(summary_path, 'w') as f:
        f.write(json.dumps(summary))

    print(f"Correct predictions: {summary['correct_predictions']}")
    print(f"Incorrect predictions: {summary['incorrect_predictions']}")
    print(f"Target label precision: {summary['target_label_precision']:.3f}")
    logger.info(f"Summary for '{class_label}': correct={summary['correct_predictions']}, incorrect={summary['incorrect_predictions']}, precision={summary['target_label_precision']:.3f}")

    return stored_samples, summary



def main():
    parser = argparse.ArgumentParser(
        description="Initialize random path worker system with aeon univariate datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available datasets
  python init_aeon_univariate.py --list-datasets

  # Custom forest parameters
  python init_aeon_univariate.py Coffee --n-estimators 100 --max-depth 5

  # Process only 5% of samples with sample percentage
  python init_aeon_univariate.py ECG200 --sample-percentage 5

  # Process only 0.05% of samples (as in your example)
  python init_aeon_univariate.py ECG200 --sample-percentage 0.05
        """
    )

    parser.add_argument('dataset_name', nargs='?',
                        help='Name of the aeon univariate dataset to load')

    parser.add_argument('--list-datasets', action='store_true',
                        help='List all available aeon univariate datasets')

    parser.add_argument('--info', action='store_true',
                        help='Show information about the dataset without processing')

    # Forest parameters
    forest_group = parser.add_argument_group('Random Forest Parameters')
    forest_group.add_argument('--n-estimators', type=int, default=50,
                              help='Number of trees in the forest (default: 50)')
    forest_group.add_argument('--criterion', type=str, choices=['gini', 'entropy'],
                              help='Split quality criterion (default: gini)')
    forest_group.add_argument('--max-depth', type=int,
                              help='Maximum depth of trees (default: None)')
    forest_group.add_argument('--min-samples-split', type=int,
                              help='Minimum samples required to split (default: 2)')
    forest_group.add_argument('--min-samples-leaf', type=int,
                              help='Minimum samples required at leaf (default: 1)')
    forest_group.add_argument('--max-features', type=str,
                              help='Number of features for best split (default: "sqrt")')
    forest_group.add_argument('--max-leaf-nodes', type=int,
                              help='Maximum number of leaf nodes (default: None)')
    forest_group.add_argument('--min-impurity-decrease', type=float,
                              help='Minimum impurity decrease for split (default: 0.0)')
    forest_group.add_argument('--bootstrap', type=str, choices=['True', 'False'],
                              help='Whether to use bootstrap samples (default: True)')
    forest_group.add_argument('--max-samples', type=float,
                              help='Fraction of samples for each tree if bootstrap=True (default: None)')
    forest_group.add_argument('--ccp-alpha', type=float,
                              help='Complexity parameter for pruning (default: 0.0)')

    # Bayesian optimization parameters
    opt_group = parser.add_argument_group('Bayesian Optimization Parameters')
    opt_group.add_argument('--optimize-rf', action='store_true',
                           help='Use Bayesian optimization to find best RF hyperparameters')
    opt_group.add_argument('--opt-n-iter', type=int, default=50,
                           help='Number of iterations for Bayesian optimization (default: 50)')
    opt_group.add_argument('--opt-cv', type=int, default=5,
                           help='Number of cross-validation folds for optimization (default: 5)')
    opt_group.add_argument('--opt-n-jobs', type=int, default=-1,
                           help='Number of parallel jobs for optimization, -1 for all cores (default: -1)')
    opt_group.add_argument('--opt-use-test', action='store_true',
                           help='Use test set for validation during optimization instead of CV. '
                                'WARNING: May lead to overfitting on test data! Use with caution.')

    # New parameters for sample percentage filtering
    parser.add_argument('--sample-percentage', type=float, default=100.0,
                        help='Process only this percentage of samples (0-100, default: 100)')
    parser.add_argument('--test-split', type=float, default=None,
                        help='If specified, combines train+test and does custom split with this test fraction (e.g., 0.3). '
                             'If not specified, uses original aeon train/test split (default: None)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--feature-prefix', type=str, default='t',
                        help='Prefix for feature names (default: "t")')
    
    # Logging parameters
    logging_group = parser.add_argument_group('Logging Parameters')
    logging_group.add_argument('--log-level', type=str, 
                              choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                              default='INFO',
                              help='Logging level (default: INFO)')
    logging_group.add_argument('--log-file', type=str, default=None,
                              help='Path to log file (default: results/init_aeon_univariate.log)')

    args = parser.parse_args()
    
    # Setup logging based on arguments
    log_level = getattr(logging, args.log_level)
    setup_logging(log_level=log_level, log_file=args.log_file)
    
    logger.info("Starting init_aeon_univariate main function")
    logger.debug(f"Parsed arguments: {args}")

    # Handle list datasets
    if args.list_datasets:
        logger.info("Listing available datasets")
        list_available_datasets()
        return 0

    # Validate arguments
    if not args.dataset_name:
        logger.error("dataset_name is required (or use --list-datasets)")
        parser.error("dataset_name is required (or use --list-datasets)")

    # Show dataset info if requested
    logger.info(f"Getting information for dataset: {args.dataset_name}")
    info = get_dataset_info(args.dataset_name)
    classes = info['classes']
    logger.debug(f"Dataset info retrieved: {info}")
    print(f"Getting information for dataset: {args.dataset_name}")
    if 'error' in info:
        logger.error(f"Error loading dataset: {info['error']}")
        print(f" Error loading dataset: {info['error']}")
        print("Make sure the dataset name is correct and aeon is installed.")
        return 1

    if args.info:
        logger.info(f"Displaying dataset information for {args.dataset_name}")
        print(f"\n Dataset Information: {args.dataset_name}")
        print(f"  Training samples: {info['train_size']}")
        print(f"  Test samples: {info['test_size']}")
        print(f"  Series length: {info['series_length']}")
        print(f"  Number of channels: {info['n_channels']}")
        print(f"  Number of classes: {info['n_classes']}")
        print(f"  Classes: {[str(c) for c in classes]}")

        return 0

    logger.info(f"Initializing Random Path Worker System for dataset: {args.dataset_name}")
    logger.info(f"Sample Percentage: {args.sample_percentage}%")
    print(f" Initializing Random Path Worker System")
    print(f" Dataset: {args.dataset_name}")
    print(f" Sample Percentage: {args.sample_percentage}%")

    try:
        # Load and prepare dataset
        logger.info(f"Loading and preparing dataset: {args.dataset_name}")
        X_train, y_train, X_test, y_test, feature_names, class_names = load_and_prepare_dataset(
            args.dataset_name, args.feature_prefix
        )
        logger.info(f"Dataset loaded successfully with {len(feature_names)} features and {len(class_names)} classes")

        # Determine which training data to use for optimization
        # If test_split is specified, we'll combine and split later in train_and_convert_forest
        # For optimization, we use the original training set
        X_train_for_opt = X_train
        y_train_for_opt = y_train
        logger.debug(f"Training data prepared: X_train shape={X_train.shape}, y_train shape={y_train.shape}")

        # Optionally optimize RF hyperparameters with Bayesian optimization
        if args.optimize_rf:
            logger.info("Starting Bayesian optimization for Random Forest hyperparameters")
            print("\n" + "="*70)
            print("BAYESIAN OPTIMIZATION MODE")
            print("="*70)

            # Get search space
            search_space = get_rf_search_space()
            logger.debug(f"Search space defined with {len(search_space)} hyperparameters")

            # Run Bayesian optimization
            logger.info(f"Running Bayesian optimization with n_iter={args.opt_n_iter}, cv={args.opt_cv}")
            best_params, best_score, test_score, optimizer = optimize_rf_hyperparameters(
                X_train_for_opt, y_train_for_opt,
                search_space=search_space,
                n_iter=args.opt_n_iter,
                cv=args.opt_cv,
                n_jobs=args.opt_n_jobs,
                random_state=args.random_state,
                verbose=1,
                X_test=X_test,
                y_test=y_test,
                use_test_for_validation=args.opt_use_test
            )
            logger.info(f"Optimization completed with best_score={best_score}, test_score={test_score}")

            # Store optimization results in DATA for future reference
            opt_results = {
                'best_params': best_params,
                'best_cv_score': best_score if not args.opt_use_test else None,
                'test_score': test_score,
                'used_test_for_validation': args.opt_use_test,
                'n_iter': args.opt_n_iter,
                'cv_folds': args.opt_cv if not args.opt_use_test else None,
                'dataset_name': args.dataset_name,
                'timestamp': datetime.datetime.now().isoformat()
            }

            opt_results_path = results_path(f"{args.dataset_name}_random_forest_optimization_results.json")
            logger.debug(f"Saving optimization results to {opt_results_path}")
            with open(opt_results_path, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                f.write(json.dumps(convert_numpy_types(opt_results)))
            logger.info("Optimization results saved successfully")

            # Use optimized parameters
            rf_params = {**best_params, 'random_state': args.random_state}
            logger.info(f"Using optimized parameters: {rf_params}")

        else:
            # Use manually specified parameters
            logger.info("Using manually specified Random Forest parameters")
            rf_params = create_forest_params(args)
            logger.debug(f"Forest parameters: {rf_params}")

        # Train and convert forest
        logger.info("Training and converting Random Forest model")
        sklearn_rf, our_forest, X_train_used, y_train_used = train_and_convert_forest(
            X_train, y_train, X_test, y_test, rf_params, feature_names,
            test_split=args.test_split, random_state=args.random_state,
            sample_percentage=args.sample_percentage
        )
        logger.info(f"Random Forest trained with {len(our_forest)} trees")

        # Store training set in DATA database
        logger.info("Storing training set to database")
        store_training_set(X_train, y_train, feature_names, args.dataset_name)

        # Store forest and endpoints
        logger.info("Storing Random Forest model")
        print("Storing Random Forest...")

        if store_forest(our_forest, f"{args.dataset_name}_random_forest.json"):
            print("Forest saved successfully")
            logger.info("Forest saved successfully")
        else:
            logger.error("Failed to save forest")
            raise Exception("Failed to save forest")

        # Extract and store feature thresholds (endpoints universe)
        logger.info("Extracting feature thresholds (endpoints universe)")
        print("Extracting feature thresholds...")
        eu_data = our_forest.extract_feature_thresholds()
        print(f"Extracted thresholds for {len(eu_data)} features")
        logger.info(f"Extracted thresholds for {len(eu_data)} features")

        logger.info("Storing endpoints universe")
        print("Storing endpoints universe...")
        if store_monotonic_dict(eu_data, f"{args.dataset_name}_endpoints_universe.json"):
            print("Endpoints universe saved successfully")
            logger.info("Endpoints universe saved successfully")
        else:
            logger.error("Failed to save endpoints universe")
            raise Exception("Failed to save endpoints universe")

        logger.info(f"Processing samples for {len(classes)} classes")
        for class_label in classes:
            # Process all test samples classified with target label
            logger.info(f"Processing samples classified as '{class_label}'")
            stored_samples, summary = process_all_classified_samples(
                dataset_name=args.dataset_name,
                class_label=class_label,
                our_forest=our_forest,
                X_test=X_test,
                y_test=y_test,
                feature_names=feature_names,
                sample_percentage=args.sample_percentage
            )
            logger.info(f"Processed {len(stored_samples)} samples for class '{class_label}'")

        print(f"\nSuccessfully completed for {args.dataset_name} with labels ", [str(c) for c in classes])
        print(f"Forest: {len(our_forest)} trees")
        print(f"Features: {len(feature_names)}")
        logger.info(f"Successfully completed initialization for {args.dataset_name}")
        logger.info(f"Forest: {len(our_forest)} trees, Features: {len(feature_names)}")

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        print("\n  Interrupted by user")
    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        print(f"\n Error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return 1

    logger.info("init_aeon_univariate main function completed successfully")
    return 0


if __name__ == "__main__":
    exit(main())
