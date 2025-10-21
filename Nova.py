import pickle
import time

class Nova:
    
    def __init__(self, path_model=None):
        self.path_model = path_model
        self.model = None

    def load_model(self, path_model):
        try:
            with open(path_model, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✅ Modelo cargado correctamente desde: {path_model}")
        
        except FileNotFoundError:
            print(f"❌ Error: No se encontró el archivo del modelo en la ruta: {path_model}")
        
        except pickle.UnpicklingError:
            print(f"❌ Error: No se pudo deserializar el modelo. El archivo puede estar corrupto o no ser un .pkl válido.")
        
        except Exception as e:
            print(f"⚠️ Error inesperado al cargar el modelo: {e}")

    def predict(self, text):
        if self.model is None:
            self.load_model(self.path_model)
        if text is None:
            return "No has ordenado nada", None, None
        
        start_time = time.time()
        predict = self.model.predict(text)
        end_time = time.time()
        probability_Max = self.model.probability(text)
        elapsed_time = end_time - start_time
        
        
        return predict, probability_Max, elapsed_time
        
            
        
        
            
            
        