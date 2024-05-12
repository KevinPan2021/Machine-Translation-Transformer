import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from nltk.translate.bleu_score import corpus_bleu
from visualization import plot_training_curves

def BLUE(ref, pred):
    ref = [[x] for x in ref]
    score = corpus_bleu(ref, pred)
    return round(score, 4)


# feed forward, no gradient is updated
@torch.no_grad
def feedforward(data_loader, model):
    # Set model to evaluation mode
    model.eval()
    
    running_loss = 0.0
    running_BLUE = 0.0
    
    criterion = nn.NLLLoss()
    device = next(model.parameters()).device

    with tqdm(total=len(data_loader)) as pbar:
        for i, (X, Y_input, Y_output) in enumerate(data_loader):
            # convert to gpu
            X = X.to(device)
            Y_input = Y_input.to(device)
            Y_output = Y_output.to(device)
            
            decoder_outputs = model(X, Y_input)
    
            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                Y_output.view(-1)
            )
            
            # update the statistic
            running_loss += loss.item()
            predicted = torch.argmax(decoder_outputs, dim=-1)
            
            blue = BLUE(Y_output.tolist(), predicted.tolist())
            running_BLUE += blue
            
            # Update tqdm description with loss and BLUE
            pbar.set_postfix({'Loss':running_loss/(i+1), 'BLUE':running_BLUE/(i+1)})
            pbar.update(1)
            
    return running_loss/len(data_loader), running_BLUE/len(data_loader)



# backpropagation for training
def backpropagation(data_loader, optimizer, model, scaler):
    # Set model to training mode
    model.train()
    
    running_loss = 0.0
    running_BLUE = 0.0
    
    criterion = nn.NLLLoss()
    device = next(model.parameters()).device
    
    with tqdm(total=len(data_loader)) as pbar:
        for i, (X, Y_input, Y_output) in enumerate(data_loader):
            
            # convert to gpu
            X = X.to(device)
            Y_input = Y_input.to(device)
            Y_output = Y_output.to(device)
            
            # mixed precision training
            with autocast(dtype=torch.float16):
                decoder_outputs = model(X, Y_input)
            
                loss = criterion(
                    decoder_outputs.view(-1, decoder_outputs.size(-1)),
                    Y_output.view(-1)
                )
                
            # update the statistic
            running_loss += loss.item()
            predicted = torch.argmax(decoder_outputs, dim=-1)
            blue = BLUE(Y_output.tolist(), predicted.tolist())
            running_BLUE += blue
            
            # Reset gradients
            optimizer.zero_grad()
    
            # Backpropagate the loss
            scaler.scale(loss).backward()
    
            # Optimization step
            scaler.step(optimizer)
    
            # Updates the scale for next iteration.
            scaler.update()
            
            # Update tqdm description with loss and BLUE
            pbar.set_postfix({'Loss': running_loss/(i+1), 'BLUE':running_BLUE/(i+1)})
            pbar.update(1)
      
    return running_loss/len(data_loader), running_BLUE/len(data_loader)



def model_training(train_data, valid_data, model):
    n_epochs = 80
    learning_rate = 1e-4
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scaler = GradScaler()
    
    # append the results of model with no training
    print(f"Epoch {0}/{n_epochs}")
    train_loss, train_blue = feedforward(train_data, model)
    valid_loss, valid_blue = feedforward(valid_data, model)
    train_losses, train_BLUEs = [train_loss], [train_blue]
    valid_losses, valid_BLUEs = [valid_loss], [valid_blue]
    
    # Early Stopping criteria
    patience = 3
    not_improved = 0
    best_valid_loss = valid_loss
    threshold = 0.01
    
    # training loop
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        train_loss, train_blue = backpropagation(train_data, optimizer, model, scaler)
        valid_loss, valid_blue = feedforward(valid_data, model)
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_BLUEs.append(train_blue)
        valid_BLUEs.append(valid_blue)
        
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
    