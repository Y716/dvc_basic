from joblib import load, dump

if __name__ == "__main__":
    # Load the trained model
    trained_model = load("models/train_model/model.joblib")
    
    # Save the trained model to the new directory
    dump(trained_model, "models/deploy_model/model-deploy.joblib")
