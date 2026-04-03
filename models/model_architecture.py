import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class FraudGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(FraudGNN, self).__init__()
        
        # Layer 1: Takes 30 features of a transaction and finds 16 hidden patterns
        self.conv1 = GCNConv(num_node_features, 16)
        
        # Layer 2: Takes the 16 hidden patterns and outputs 2 classes (Fraud or Normal)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        # x = node features (the transaction details)
        # edge_index = the connections we made in Step 5
        x, edge_index = data.x, data.edge_index

        # Step A: Pass through first layer
        x = self.conv1(x, edge_index)
        x = F.relu(x) # ReLU helps the AI learn better
        
        # Step B: Pass through second layer
        x = self.conv2(x, edge_index)
        
        # Step C: Output the final probabilities for the 2 classes
        return F.log_softmax(x, dim=1)

# A small test to make sure our code has no errors
if __name__ == "__main__":
    # We have 30 features in our dataset, and 2 classes (0 or 1)
    model = FraudGNN(num_node_features=30, num_classes=2)
    print("✅ GNN Model architecture created successfully!")
    print(model)