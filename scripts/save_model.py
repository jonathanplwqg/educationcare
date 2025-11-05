"""
Helper script to save your trained model and preprocessors from the Final.ipynb notebook.

Run this after training your model in the notebook to save the necessary files for the Streamlit app.

Usage:
    1. In your Final.ipynb notebook, after training your model, run:
       
       import pickle
       from save_model import save_model_artifacts
       
       # Assuming you have these variables in your notebook:
       # - best_model or final_model (your trained classifier)
       # - scaler (StandardScaler or PowerTransformer)
       # - encoder (OneHotEncoder)
       # - original_features (list of feature names)
       # - target_encoder (LabelEncoder for target variable)
       
       save_model_artifacts(
           model=best_model,  # or your trained model variable
           scaler=scaler,
           encoder=encoder,
           feature_names=original_features,
           target_encoder=target_encoder,
           cluster_model=None,  # optional: your KMeans model if you want cluster predictions
           umap_reducer=None    # optional: your UMAP model if you want embeddings
       )
    
    2. Run the Streamlit app:
       streamlit run app.py
"""

import pickle
import json
from pathlib import Path

def save_model_artifacts(
    model,
    scaler,
    encoder,
    feature_names,
    target_encoder,
    cluster_model=None,
    umap_reducer=None,
    output_dir="."
):
    """
    Save all model artifacts needed for the Streamlit app.
    
    Parameters:
    -----------
    model : sklearn classifier
        Trained classification model (e.g., RandomForestClassifier, XGBoost, etc.)
    scaler : sklearn scaler
        Fitted scaler (StandardScaler, PowerTransformer, etc.)
    encoder : sklearn encoder
        Fitted OneHotEncoder for categorical variables
    feature_names : list
        List of feature names in the order expected by the model
    target_encoder : LabelEncoder
        Encoder for the target variable (final_result)
    cluster_model : KMeans, optional
        Trained clustering model for student personas
    umap_reducer : UMAP, optional
        Fitted UMAP reducer for dimensionality reduction
    output_dir : str
        Directory to save the artifacts
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("💾 Saving model artifacts...")
    
    # Save the main classification model
    with open(output_path / 'model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ Saved model.pkl")
    
    # Save the scaler
    with open(output_path / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✅ Saved scaler.pkl")
    
    # Save the encoder
    with open(output_path / 'encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    print("✅ Saved encoder.pkl")
    
    # Save the target encoder
    with open(output_path / 'target_encoder.pkl', 'wb') as f:
        pickle.dump(target_encoder, f)
    print("✅ Saved target_encoder.pkl")
    
    # Save feature names
    with open(output_path / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)
    print("✅ Saved feature_names.json")
    
    # Save optional models
    if cluster_model is not None:
        with open(output_path / 'cluster_model.pkl', 'wb') as f:
            pickle.dump(cluster_model, f)
        print("✅ Saved cluster_model.pkl")
    
    if umap_reducer is not None:
        with open(output_path / 'umap_reducer.pkl', 'wb') as f:
            pickle.dump(umap_reducer, f)
        print("✅ Saved umap_reducer.pkl")
    
    # Save metadata
    metadata = {
        'model_type': type(model).__name__,
        'scaler_type': type(scaler).__name__,
        'encoder_type': type(encoder).__name__,
        'n_features': len(feature_names),
        'target_classes': target_encoder.classes_.tolist() if hasattr(target_encoder, 'classes_') else None,
        'has_cluster_model': cluster_model is not None,
        'has_umap_reducer': umap_reducer is not None
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("✅ Saved metadata.json")
    
    print("\n🎉 All artifacts saved successfully!")
    print(f"📁 Location: {output_path.absolute()}")
    print("\n📋 Files created:")
    print("   - model.pkl (classification model)")
    print("   - scaler.pkl (feature scaler)")
    print("   - encoder.pkl (categorical encoder)")
    print("   - target_encoder.pkl (target variable encoder)")
    print("   - feature_names.json (feature names)")
    print("   - metadata.json (model metadata)")
    if cluster_model is not None:
        print("   - cluster_model.pkl (clustering model)")
    if umap_reducer is not None:
        print("   - umap_reducer.pkl (UMAP reducer)")
    
    print("\n🚀 You can now run: streamlit run app.py")


# Example usage for your notebook:
if __name__ == "__main__":
    print("""
    This is a helper module. Import it in your Final.ipynb notebook:
    
    # In your notebook, after training:
    from save_model import save_model_artifacts
    
    # Example with RandomForest from your results_comparison:
    # Get the best model from your comparison
    best_strategy = results_comparison[0]  # Assuming first is best
    best_model = best_strategy['Model']
    
    save_model_artifacts(
        model=best_model,
        scaler=scaler,  # Your PowerTransformer
        encoder=encoder,  # Your OneHotEncoder
        feature_names=original_features,  # or selected_cluster_features
        target_encoder=target_encoder,  # Your LabelEncoder for final_result
        cluster_model=kmeans,  # Your best KMeans model from UMAP analysis
        umap_reducer=umap_reducer  # Your UMAP model
    )
    """)
