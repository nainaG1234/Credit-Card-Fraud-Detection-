import torch
import torch.nn.functional as F
from models.model_architecture import FraudGNN

def train_model():
    print("Loading graph data...")
    # FIX: Added weights_only=False to bypass PyTorch 2.6 security restriction
    data = torch.load('data/graph_data.pt', weights_only=False)
    
    print("Initializing GNN Model...")
    # 30 features, 2 classes (0: Normal, 1: Fraud)
    model = FraudGNN(num_node_features=30, num_classes=2)
    
    # Optimizer (Adam) acts like a teacher correcting the AI's mistakes
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    print("Starting training...\n")
    model.train() # Put the model in training mode
    
    for epoch in range(1, 201): # Train for 200 loops
        optimizer.zero_grad()      # Clear old memory
        out = model(data)          # AI makes its guesses
        
        # Calculate loss (how wrong the AI is)
        loss = F.nll_loss(out, data.y)
        
        loss.backward()            # AI realizes its mistakes
        optimizer.step()           # AI updates its brain to do better next time
        
        # Calculate accuracy
        pred = out.argmax(dim=1)   # Get the final 0 or 1 prediction
        correct = (pred == data.y).sum()
        acc = int(correct) / int(data.num_nodes)
        
        # Print progress every 20 loops
        if epoch % 20 == 0:
            print(f'Epoch: {epoch:03d} | Loss: {loss:.4f} | Accuracy: {acc:.4f}')
            
    print("\n✅ Training completed!")
    
    # Save the smart AI brain to a file
    save_path = 'models/fraud_model.pth'
    torch.save(model.state_dict(), save_path)
    print(f"✅ Trained model saved successfully to: {save_path}")

if __name__ == "__main__":
    train_model()