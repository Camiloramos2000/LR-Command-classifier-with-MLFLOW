import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
from Model import ModelLR
from Data import Data
import pandas as pd
import uuid
from DecodedPipelineWrapper import DecodedPipelineWrapper




# ==========================================================
# 🔍 Helper function: Display model version information
# ==========================================================
def print_model_version_info(mv):
    print("\n📋 CURRENT MODEL INFORMATION")
    print("=" * 80)
    mv = client.get_model_version(name=mv.name, version=mv.version)
    print(f"🧾 Name: {mv.name}")
    print(f"🔢 Version: {mv.version}")
    print(f"📝 Description: {mv.description}")
    print(f"📦 Stage: {mv.current_stage}\n")
    

def set_stage_and_alias(client, name, version, stage):
    client.transition_model_version_stage(
        name=name,
        version=version,
        stage=stage,
        archive_existing_versions=False
    )

    client.set_registered_model_alias(
        name=name,
        alias=stage,
        version=version
    )


# ==========================================================
# 🚀 MAIN ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🌍  STARTING MLFLOW PROCESS")
    print("=" * 80 + "\n")

    # -------------------------------------------
    # 🌐 MLFLOW CONFIGURATION
    # -------------------------------------------
    print(f"🔗 Current Tracking URI: {mlflow.get_tracking_uri()}\n")

    mlflow.set_experiment("command_classifier_model")
    run_name = f"C.C.M.LR-Run-{uuid.uuid4()}"

    with mlflow.start_run(run_name=run_name) as run:
        client = MlflowClient()

        # -------------------------------------------
        # ⚙️ MODEL PARAMETERS
        # -------------------------------------------
        params = {
            "C":10.0,                     # Menos regularización → más capacidad de aprender
            "penalty":"l2",               # Regularización estándar
            "solver":"lbfgs",             # Robusto y eficiente para multiclase
            "multi_class":"auto",         # Detecta automáticamente
            "max_iter":2000,              # Aumenta iteraciones para convergencia estable
            "class_weight":"balanced",    # Compensa si tienes clases desbalanceadas
            "n_jobs":-1,                  # Usa todos los núcleos disponibles
            "random_state": 42
        }

        modellr = ModelLR(parameters=params)
        data = Data()

        # -------------------------------------------
        # 🧠 MODEL TRAINING
        # -------------------------------------------
        pipeline, accuracy, precision, recall, f1, elapsed_time = modellr.train()

        print("\n📊 Logging metrics and parameters to MLflow...\n")

        # -------------------------------------------
        # 🪣 LOGGING PARAMETERS AND METRICS
        # -------------------------------------------
        mlflow.log_params(params)
        mlflow.log_param("transformer", "Sentence transformers")
        mlflow.log_artifact("commands_dataset.csv")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1)

        print("✅ Model trained and evaluated successfully.")
        print("📁 Metrics, parameters, and dataset logged to MLflow.\n")
        print("-" * 80 + "\n")

        # -------------------------------------------
        # 📦 MODEL REGISTRATION IN MLFLOW MODEL REGISTRY
        # -------------------------------------------
        print("=" * 80)
        print("📦 REGISTERING MODEL IN MLFLOW MODEL REGISTRY")
        print("=" * 80 + "\n")

        client = MlflowClient()
        model_uri = f"runs:/{run.info.run_id}/{modellr.name}"
        
        #input de ejemplo       
        order = "Apagar el computador"
        prediction, prob_max, elapsed_time = modellr.test(order)

        input_example = data.X
        signature = infer_signature(data.X, prediction)
        tag = {
            "version": "1.0",
            "type_model": "Logistic Regression",
            "created_by": "Camilo Ramos"
        }
        
        encoder_fitted = modellr.data.encoder
        
        decoded_pipeline = DecodedPipelineWrapper(modellr.pipeline, encoder_fitted)
        
        

        model_info = mlflow.sklearn.log_model(
            name=modellr.name,
            sk_model=decoded_pipeline,
            input_example=input_example,
            signature=signature,
            tags=tag
        )

        try:
            mv = client.create_registered_model(
                name=modellr.name,
                description="Command classification model using Logistic Regression"
            )
            print("📦✅ Model successfully registered in MLflow Model Registry.\n")

        except mlflow.exceptions.MlflowException as e:
            if "already exists" in str(e):
                print(f"⚠️ Model '{modellr.name}' already exists. Using the existing one.\n")
                mv = client.get_registered_model(modellr.name)
            else:
                raise e

        print("-" * 80 + "\n")
        print("🆕 Creating a new model version...")

        try:
            mv = client.create_model_version(
                name=modellr.name,
                source=model_info.model_uri,
                run_id=run.info.run_id,
                tags={"version":3}
            )
        except mlflow.exceptions.MlflowException as e:
            if "already exists" in str(e):
                print(f"⚠️ Version already exists. Using the current one.\n")
                mv = client.get_latest_versions(modellr.name)[-1]
            else:
                raise e

        mv = client.update_model_version(
            name=mv.name,
            version=mv.version,
            description="Initial version of the command classifier using Logistic Regression"
        )
        
        set_stage_and_alias(client, name=mv.name, version=mv.version, stage="Staging")

        print("📦✅ Model version successfully created.\n")
        print("-" * 80 + "\n")

        # -------------------------------------------
        # 🔍 MODEL INFORMATION
        # -------------------------------------------
        print_model_version_info(mv)

        # -------------------------------------------
        # 🚦 TRANSITION TO PRODUCTION
        # -------------------------------------------
        print("=" * 80)
        print("🚀 TRANSITIONING MODEL TO STAGE: PRODUCTION")
        print("=" * 80 + "\n")

        print("🚦 Transitioning model to 'Production' stage...")
        
        set_stage_and_alias(client, name=mv.name, version=mv.version, stage="Production")

        print("✅ Model successfully moved to 'Production' stage.\n")
        print("-" * 80 + "\n")

        print_model_version_info(mv)

        print("=" * 80)
        print("🎯 FULL PROCESS COMPLETED SUCCESSFULLY")
        print("=" * 80 + "\n")

