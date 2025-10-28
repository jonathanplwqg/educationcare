# Add this cell at the end of your Final.ipynb notebook to save the model for the Streamlit app

"""
SAVE MODEL FOR STREAMLIT APP
=============================
Run this cell after training your model to save all artifacts needed for the Streamlit app.
"""

import pickle
import json
from pathlib import Path

def save_model_for_streamlit():
    """Save all model artifacts for the Streamlit app"""
    
    print("💾 Saving model artifacts for Streamlit app...")
    
    # Get the best model from your classification results
    # Adjust these variable names based on your actual notebook variables
    
    # Option 1: If you used results_comparison
    if 'results_comparison' in globals():
        best_model = results_comparison[0]['Model']  # First result is usually best
        print(f"✅ Using model from results_comparison: {type(best_model).__name__}")
    
    # Option 2: If you have a specific model variable
    elif 'best_model' in globals():
        print(f"✅ Using best_model: {type(best_model).__name__}")
    
    # Option 3: Use any trained model
    elif 'xgb_model' in globals():
        best_model = xgb_model
        print(f"✅ Using xgb_model")
    elif 'lgb_model' in globals():
        best_model = lgb_model
        print(f"✅ Using lgb_model")
    else:
        print("❌ No model found! Train a model first.")
        return
    
    # Save model
    with open('model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print("✅ Saved model.pkl")
    
    # Save scaler
    if 'scaler' in globals():
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        print("✅ Saved scaler.pkl")
    else:
        print("⚠️  No scaler found")
    
    # Save encoder
    if 'encoder' in globals():
        with open('encoder.pkl', 'wb') as f:
            pickle.dump(encoder, f)
        print("✅ Saved encoder.pkl")
    else:
        print("⚠️  No encoder found")
    
    # Save target encoder (create if doesn't exist)
    from sklearn.preprocessing import LabelEncoder
    target_encoder = LabelEncoder()
    target_encoder.fit(['Pass', 'Fail', 'Distinction', 'Withdrawn'])
    
    with open('target_encoder.pkl', 'wb') as f:
        pickle.dump(target_encoder, f)
    print("✅ Saved target_encoder.pkl")
    
    # Save feature names
    if 'original_features' in globals():
        feature_list = original_features
    elif 'selected_cluster_features' in globals():
        feature_list = selected_cluster_features
    else:
        feature_list = [col for col in final_data.columns if col != 'final_result']
    
    with open('feature_names.json', 'w') as f:
        json.dump(feature_list, f, indent=2)
    print(f"✅ Saved feature_names.json ({len(feature_list)} features)")
    
    # Save cluster model (optional)
    if 'best_result' in globals() and 'kmeans' in best_result:
        with open('cluster_model.pkl', 'wb') as f:
            # Note: best_result from UMAP analysis contains the KMeans model
            # You might need to retrain a simple KMeans on the embedding
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=best_result['best_k'], random_state=42)
            kmeans.fit(best_result['embedding'])
            pickle.dump(kmeans, f)
        print("✅ Saved cluster_model.pkl")
    
    # Save UMAP reducer (optional)
    if 'best_result' in globals() and 'umap_reducer' in best_result:
        with open('umap_reducer.pkl', 'wb') as f:
            pickle.dump(best_result['umap_reducer'], f)
        print("✅ Saved umap_reducer.pkl")
    
    # Save metadata
    metadata = {
        'model_type': type(best_model).__name__,
        'n_features': len(feature_list),
        'target_classes': ['Pass', 'Fail', 'Distinction', 'Withdrawn'],
        'created_date': str(pd.Timestamp.now()),
        'notebook': 'Final.ipynb'
    }
    
    with open('metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("✅ Saved metadata.json")
    
    print("\n" + "="*60)
    print("🎉 All artifacts saved successfully!")
    print("="*60)
    print("\n📋 Files created:")
    print("   ✓ model.pkl")
    print("   ✓ scaler.pkl")
    print("   ✓ encoder.pkl")
    print("   ✓ target_encoder.pkl")
    print("   ✓ feature_names.json")
    print("   ✓ metadata.json")
    print("   ✓ cluster_model.pkl (optional)")
    print("   ✓ umap_reducer.pkl (optional)")
    
    print("\n🚀 Next steps:")
    print("   1. Open terminal in this directory")
    print("   2. Run: streamlit run app.py")
    print("   3. Test the app in your browser!")
    
    return True

# Run the function
save_model_for_streamlit()
