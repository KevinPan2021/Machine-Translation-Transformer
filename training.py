import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from visualization import plot_training_curves
import os

# Function to calculate BLUE score
def BLUE(predictions, references):
    # Convert predictions and references to lists of strings
    predictions = [[str(tok) for tok in pred] for pred in predictions]
    references = [[[str(tok) for tok in ref]] for ref in references]

    # Calculate BLEU score
    return corpus_bleu(references, predictions)



# feed forward, no gradient is updated
def feedforward(data_loader, model, criterion, device):
    epoch_loss = 0.0
    all_predictions = []
    all_references = []
    
    model.eval()
    for X, Y_input, Y_output in data_loader:
        
        # convert to gpu
        X = X.to(device)
        Y_input = Y_input.to(device)
        Y_output = Y_output.to(device)
        
        decoder_outputs = model(X, Y_input)

        
        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            Y_output.view(-1)
        )

        epoch_loss += loss.item()
        
        # Collect predictions and references for BLUE score calculation
        all_predictions.extend(torch.argmax(decoder_outputs, dim=-1).cpu().tolist())
        all_references.extend(Y_output.cpu().tolist())
        
    epoch_loss /= len(data_loader)
    
    # Calculate BLUE score
    epoch_BLUE = BLUE(all_predictions, all_references)
    
    return epoch_loss, epoch_BLUE


# backpropagation for training
def backpropagation(data_loader, optimizer, model, criterion, device):
    epoch_loss = 0.0
    all_predictions = []
    all_references = []
    
    model.train()
    for X, Y_input, Y_output in tqdm(data_loader):
        
        # convert to gpu
        X = X.to(device)
        Y_input = Y_input.to(device)
        Y_output = Y_output.to(device)
        
        model.zero_grad()
    
        decoder_outputs = model(X, Y_input)
    
        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            Y_output.view(-1)
        )
        loss.backward()
    
        optimizer.step()
    
        epoch_loss += loss.item()
        
        # Collect predictions and references for BLUE score calculation
        all_predictions.extend(torch.argmax(decoder_outputs, dim=-1).cpu().tolist())
        all_references.extend(Y_output.cpu().tolist())
        
    epoch_loss /= len(data_loader)
    
    # Calculate BLUE score
    epoch_BLUE = BLUE(all_predictions, all_references)
    
    return epoch_loss, epoch_BLUE



def model_training(train_data, valid_data, model, device):
    n_epochs = 100
    learning_rate=0.001
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if f'{type(model).__name__.lower()}_optimizer.pth' in os.listdir():
        optimizer.load_state_dict(torch.load(f'{type(model).__name__}_optimizer.pth'))
    criterion = nn.NLLLoss()
    
    # append the results of model with no training
    train_loss, train_blue = feedforward(train_data, model, criterion, device)
    valid_loss, valid_blue = feedforward(valid_data, model, criterion, device)
    print(f"Epoch 0/{n_epochs} | Train BLUE: {train_blue:.3f} | Train Loss: {train_loss:.3f} | Valid BLUE: {valid_blue:.3f} | Valid Loss: {valid_loss:.3f}")
    
    
    # create lists to keep track
    train_losses = [train_loss]
    valid_losses = [valid_loss]
    train_BLUEs = [train_blue]
    valid_BLUEs = [valid_blue]
    
    # Early Stopping criteria
    patience = 3
    not_improved = 0
    best_valid_loss = valid_loss
    threshold = 0.01
    
    # training loop
    for epoch in range(n_epochs):
        train_loss, train_blue = backpropagation(train_data, optimizer, model, criterion, device)
        valid_loss, valid_blue = feedforward(valid_data, model, criterion, device)
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_BLUEs.append(train_blue)
        valid_BLUEs.append(valid_blue)
        
        print(f"Epoch {epoch+1}/{n_epochs} | Train BLUE: {train_blue:.3f} | Train Loss: {train_loss:.3f} | Valid BLUE: {valid_blue:.3f} | Valid Loss: {valid_loss:.3f}")
        
        # evaluate the current preformance
        # strictly better
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            not_improved = 0
            # save the best model based on validation loss
            torch.save(model.state_dict(), f'{type(model).__name__}.pth')
            # also save the optimizer state for future training
            torch.save(optimizer.state_dict(), f'{type(model).__name__}_optimizer.pth')

        # becomes worst
        elif valid_loss > best_valid_loss + threshold:
            not_improved += 1
            if not_improved >= patience:
                print('Early Stopping Activated')
                break
            
    plot_training_curves(train_BLUEs, train_losses, valid_BLUEs, valid_losses)
    