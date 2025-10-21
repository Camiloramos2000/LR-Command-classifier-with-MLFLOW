from Nova import Nova
import time
import os

if __name__ == "__main__":
    os.system("clear" if os.name == "posix" else "cls")  # Limpia la consola
    
    print("\n" + "=" * 80)
    print("🌌  WELCOME TO NOVA'S COMMAND CLASSIFIER  🌌".center(80))
    print("=" * 80 + "\n")
    
    model = Nova(path_model="model.pkl")
    model.load_model(model.path_model)
    
    while True:
        
        # 📝 Input
        print("\n" + "-" * 80 + "\n")
        order = input("👉 Enter a command: ")
        
        if order.lower() == "exit":
            break
        

        # 🕒 Run prediction
        start_time = time.time()
        prediction, prob_max, elapsed_time = model.predict(order)
        end_time = time.time()
        
        print("\n" + "-" * 80)

        
        # 🎯 Output section
        print(f"🧠 Detected Command  :  {prediction}")
        print(f"📈 Confidence Level  :  {prob_max}%")
        print(f"⏱️  Processing Time   :  {elapsed_time:.4f} seconds\n")
    
    print("=" * 80)
    print("✨ Thank you for using Nova! ✨".center(80))
    print("=" * 80 + "\n")
