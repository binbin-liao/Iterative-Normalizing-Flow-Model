import torch
from tools import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from IPython.display import clear_output

OUTPUT_SIZE = 152
INPUT_SIZE = 186
BATCH_SIZE = 256

loss_history = []

def train_obs_normalization(Obs_Normal_Flow,opt_obs,train_data,var_np,test_data,label=True,epochs=10001):
    train_loss_history=[]
    test_loss_history=[]
    err_dist = torch.distributions.MultivariateNormal(torch.zeros(INPUT_SIZE,dtype=torch.float32),
                                                                torch.diag(torch.from_numpy(np.power(var_np,2))).
                                                                    type(torch.float32))
    for it in tqdm(range(epochs)):
        Obs_Normal_Flow.train(True)
        if label == True:
            obs_sample,ert_sample,_ = next(iter(train_data))
        else:
            obs_sample,ert_sample = next(iter(train_data))
        error = err_dist.sample((BATCH_SIZE,)).cuda()
        obs_sample = obs_sample.to(device)+error
        opt_obs.zero_grad()
        z,log_jac_det = Obs_Normal_Flow(obs_sample)
        loss = 0.5*torch.sum(z**2,1)-log_jac_det
        loss = loss.mean()/INPUT_SIZE
        loss.backward()
        opt_obs.step()
        opt_obs.zero_grad()

        if it%100 == 0:
            train_loss_history.append(loss.item())
            Obs_Normal_Flow.eval()
            clear_output(wait=True)
            N1 = 66
            N2 = 99
            if label == True:
                obs_sample,ert_sample,_ = next(iter(test_data))
            else:
                obs_sample,ert_sample = next(iter(test_data))
            error = err_dist.sample((BATCH_SIZE,)).cuda()
            obs_sample = obs_sample.to(device)+error
            z,log_jac_det = Obs_Normal_Flow(obs_sample)
            loss = 0.5*torch.sum(z**2,1)-log_jac_det
            loss = loss.mean()/INPUT_SIZE
            test_loss_history.append(loss.item())
            
            rand_data = torch.randn(BATCH_SIZE,INPUT_SIZE,device=device)
            obs_fake,_ = Obs_Normal_Flow(rand_data,rev=True,jac=False)
            X,Y = obs_sample.detach().cpu().numpy(), obs_fake.detach().cpu().numpy()

            fig, axes = plt.subplots(1,2, figsize=(12, 6), dpi=100, sharex=True, sharey=True)
            axes[0].scatter(X[:,N1],X[:,N2],edgecolors='black')
            axes[1].scatter(Y[:,N1],Y[:,N2],edgecolors='black')
            plt.xlim(-2.,2.)
            plt.ylim(-2.,2.)

            fig.tight_layout()
            torch.cuda.empty_cache(); gc.collect()
            plt.show(fig); plt.close(fig)

            fig, ax = plt.subplots(1, 1, figsize=(12, 3), dpi=100)
            ax.set_title('loss', fontsize=12)
            ax.plot(ewma(train_loss_history), c='red', label='Train_loss')
            ax.plot(ewma(test_loss_history), c='blue', label='Test_loss')
            fig.tight_layout()
            plt.show(fig); plt.close(fig)  

def train_mod_normalization(Mod_Normal_Flow,opt_mod,train_data,var_np,test_data,label=True,epochs=10001):
    train_loss_history = []
    test_loss_history = []

    for it in tqdm(range(epochs)):
        Mod_Normal_Flow.train(True)
        if label == True:
            obs_sample,ert_sample,weight = next(iter(train_data))
        else:
            obs_sample,ert_sample = next(iter(train_data))
        ert_sample = ert_sample.to(device)
        opt_mod.zero_grad()
        z,log_jac_det = Mod_Normal_Flow(ert_sample)
        loss = 0.5*torch.sum(z**2,1)-log_jac_det
        #if label:
            #loss = loss * weight.to(device)
        loss = loss.mean()/OUTPUT_SIZE
        loss.backward()
        opt_mod.step()
        opt_mod.zero_grad()
        if it%100 == 0:
            train_loss_history.append(loss.item())
            Mod_Normal_Flow.eval()
            clear_output(wait=True)
            N1 = model_pair['Lm1_den']
            N2 = model_pair['Lm1_vpv']
            if label == True:
                obs_sample,ert_sample,weight = next(iter(test_data))
            else:
                obs_sample,ert_sample = next(iter(test_data))
            ert_sample = ert_sample.to(device)
            z,log_jac_det = Mod_Normal_Flow(ert_sample)
            loss = 0.5*torch.sum(z**2,1)-log_jac_det
            #if label:
                #loss = loss * weight.to(device)
            loss = loss.mean()/OUTPUT_SIZE
            test_loss_history.append(loss.item())
            
            rand_data = torch.randn(BATCH_SIZE,OUTPUT_SIZE,device=device)
            ert_fake,_ = Mod_Normal_Flow(rand_data,rev=True,jac=False)
            X,Y = ert_sample.detach().cpu().numpy(), ert_fake.detach().cpu().numpy()

            fig, axes = plt.subplots(1,2, figsize=(12, 6), dpi=100, sharex=True, sharey=True)
            axes[0].scatter(X[:,N1],X[:,N2],edgecolors='black')
            axes[1].scatter(Y[:,N1],Y[:,N2],edgecolors='black')

            fig.tight_layout()
            torch.cuda.empty_cache(); gc.collect()
            plt.show(fig); plt.close(fig)

            fig, ax = plt.subplots(1, 1, figsize=(12, 3), dpi=100)
            ax.set_title('loss', fontsize=12)
            ax.plot(ewma(train_loss_history), c='red', label='Train_loss')
            ax.plot(ewma(test_loss_history), c='blue', label='Test_loss')
            fig.tight_layout()
            plt.show(fig); plt.close(fig)

            
def train_inverse_flow(Obs_Normal_Flow,Mod_Normal_Flow,Inverse_flow,opt_inv,train_data,var_np,obs_tensor,test_data,label=True,epochs=10001):
    train_loss_history = []
    test_loss_history = []
    #loss2_history = []
    Obs_Normal_Flow.eval();
    Mod_Normal_Flow.eval();
    err_dist = torch.distributions.MultivariateNormal(torch.zeros(INPUT_SIZE,dtype=torch.float32),
                                                                    torch.diag(torch.from_numpy(np.power(var_np,2))).
                                                                    type(torch.float32))
    for it in tqdm(range(epochs)):
        error = err_dist.sample((BATCH_SIZE,)).cuda()
        if label == True:
            obs_sample,ert_sample, weight = next(iter(train_data))
            obs_sample, ert_sample, weight = obs_sample.to(device), ert_sample.to(device),weight.to(device)
        else:
            obs_sample,ert_sample = next(iter(train_data))
            obs_sample, ert_sample = obs_sample.to(device), ert_sample.to(device)
            
        obs_code = Obs_Normal_Flow(obs_sample+error)[0].detach()
        ert_code = Mod_Normal_Flow(ert_sample)[0].detach()

        Inverse_flow.train()
        opt_inv.zero_grad()
        z,log_jac_det = Inverse_flow(ert_code,obs_code)
        if label == True:
            #loss = (0.5*torch.sum(z**2,1)-log_jac_det)*weight
            loss = (0.5*torch.sum(z**2,1)-log_jac_det)
        else:
            loss = (0.5*torch.sum(z**2,1)-log_jac_det)
        loss = loss.mean()/OUTPUT_SIZE

        #rand_ert = torch.randn(BATCH_SIZE,OUTPUT_SIZE,device=device)
        #ert_fake_code = Inverse_flow(rand_ert,obs_code,rev=True,jac=False)[0].detach()
        #ert_fake = Mod_Normal_Flow(ert_fake_code,rev=True,jac=False)[0].detach()
        #loss2 = ((ert_fake-ert_sample)**2).mean()
        #loss = loss+loss2*10.

        loss.backward()
        opt_inv.step()
        opt_inv.zero_grad()
        if it%100 == 0:
            train_loss_history.append(loss.item())
            #train_loss_history.append((loss-loss2*10.).item())
            Inverse_flow.eval()
            clear_output(wait=True)
            error = err_dist.sample((BATCH_SIZE,)).cuda()
            
            if label == True:
                obs_sample,ert_sample,_ = next(iter(test_data))
            else:
                obs_sample,ert_sample = next(iter(test_data))
            obs_sample, ert_sample = obs_sample.to(device), ert_sample.to(device)
            obs_code = Obs_Normal_Flow(obs_sample+error)[0].detach()
            ert_code = Mod_Normal_Flow(ert_sample)[0].detach()

            z,log_jac_det = Inverse_flow(ert_code,obs_code)
            if label == True:
                #loss = (0.5*torch.sum(z**2,1)-log_jac_det)*weight
                loss = (0.5*torch.sum(z**2,1)-log_jac_det)
            else:
                loss = (0.5*torch.sum(z**2,1)-log_jac_det)
            loss = loss.mean()/OUTPUT_SIZE
            test_loss_history.append(loss.item())

            #rand_ert = torch.randn(BATCH_SIZE,OUTPUT_SIZE,device=device)
            #ert_fake_code = Inverse_flow(rand_ert,obs_code,rev=True,jac=False)[0].detach()
            #ert_fake = Mod_Normal_Flow(ert_fake_code,rev=True,jac=False)[0].detach()
            #loss2 = ((ert_fake-ert_sample)**2).mean()
            #loss2_history.append(loss2.item())

            rand_ert = torch.randn(BATCH_SIZE,OUTPUT_SIZE,device=device)
            ert_fake_code = Inverse_flow(rand_ert,obs_code,rev=True,jac=False)[0].detach()
            ert_fake = Mod_Normal_Flow(ert_fake_code,rev=True,jac=False)[0].detach()

            X,Y = ert_sample.cpu().numpy(),ert_fake.cpu().numpy()
            fig, axes = plt.subplots(2,2, figsize=(12, 12), dpi=100, sharex=True, sharey=True)
            N1 = model_pair['Lm1_den']
            N2 = model_pair['Lm1_vpv']
            #N1 = 12
            #N2 = 82
            axes[0,0].scatter(X[:, N1], X[:, N2],edgecolors='black')
            axes[0,1].scatter(Y[:, N1], Y[:, N2],edgecolors='black')

            obs_code = Obs_Normal_Flow(obs_tensor)[0].detach()
            ert_fake_code = Inverse_flow(rand_ert,obs_code,rev=True,jac=False)[0].detach()
            ert_fake = Mod_Normal_Flow(ert_fake_code,rev=True,jac=False)[0].detach()
            ert_fake = ert_fake.cpu().numpy()
            axes[1,1].scatter(ert_fake[:, N1], ert_fake[:, N2],edgecolors='black')

            fig.tight_layout()
            torch.cuda.empty_cache(); gc.collect()
            plt.show(fig); plt.close(fig)

            #画w2
            fig, ax = plt.subplots(1, 1, figsize=(12, 3), dpi=100)
            ax.set_title('loss', fontsize=12)
            ax.plot(ewma(train_loss_history), c='red', label='train_loss')
            ax.plot(ewma(test_loss_history), c='blue', label='test_loss')
            #ax.plot(ewma(loss2_history),c='green',label='loss2')
            fig.tight_layout()
            plt.show(fig); plt.close(fig) 

def train_inverse_flow_original2(Mod_Normal_Flow,Inverse_flow,opt_inv,train_data,var_np,obs_tensor,test_data,label=True,epochs=10001):
    train_loss_history = []
    test_loss_history = []
    #loss2_history = []
    Mod_Normal_Flow.eval();
    err_dist = torch.distributions.MultivariateNormal(torch.zeros(INPUT_SIZE,dtype=torch.float32),
                                                                    torch.diag(torch.from_numpy(np.power(var_np,2))).
                                                                    type(torch.float32))
    for it in tqdm(range(epochs)):
        error = err_dist.sample((BATCH_SIZE,)).cuda()
        if label == True:
            obs_sample, ert_sample, weight = next(iter(train_data))
            obs_sample, ert_sample, weight = obs_sample.to(device), ert_sample.to(device),weight.to(device)
        else:
            obs_sample, ert_sample = next(iter(train_data))
            obs_sample, ert_sample = obs_sample.to(device), ert_sample.to(device)

        ert_code = Mod_Normal_Flow(ert_sample)[0].detach()

        Inverse_flow.train()
        opt_inv.zero_grad()
        z,log_jac_det = Inverse_flow(ert_code,obs_sample)
        if label == True:
            #loss = (0.5*torch.sum(z**2,1)-log_jac_det)*weight
            loss = (0.5*torch.sum(z**2,1)-log_jac_det)
        else:
            loss = (0.5*torch.sum(z**2,1)-log_jac_det)
        loss = loss.mean()/OUTPUT_SIZE

        #rand_ert = torch.randn(BATCH_SIZE,OUTPUT_SIZE,device=device)
        #ert_fake_code = Inverse_flow(rand_ert,obs_code,rev=True,jac=False)[0].detach()
        #ert_fake = Mod_Normal_Flow(ert_fake_code,rev=True,jac=False)[0].detach()
        #loss2 = ((ert_fake-ert_sample)**2).mean()
        #loss = loss+loss2*10.

        loss.backward()
        opt_inv.step()
        opt_inv.zero_grad()
        if it%100 == 0:
            train_loss_history.append(loss.item())
            #train_loss_history.append((loss-loss2*10.).item())
            Inverse_flow.eval()
            clear_output(wait=True)
            error = err_dist.sample((BATCH_SIZE,)).cuda()
            
            if label == True:
                obs_sample,ert_sample,_ = next(iter(test_data))
            else:
                obs_sample,ert_sample = next(iter(test_data))
            obs_sample, ert_sample = obs_sample.to(device), ert_sample.to(device)
            ert_code = Mod_Normal_Flow(ert_sample)[0].detach()

            z,log_jac_det = Inverse_flow(ert_code,obs_sample)
            if label == True:
                #loss = (0.5*torch.sum(z**2,1)-log_jac_det)*weight
                loss = (0.5*torch.sum(z**2,1)-log_jac_det)
            else:
                loss = (0.5*torch.sum(z**2,1)-log_jac_det)
            loss = loss.mean()/OUTPUT_SIZE
            test_loss_history.append(loss.item())

            #rand_ert = torch.randn(BATCH_SIZE,OUTPUT_SIZE,device=device)
            #ert_fake_code = Inverse_flow(rand_ert,obs_code,rev=True,jac=False)[0].detach()
            #ert_fake = Mod_Normal_Flow(ert_fake_code,rev=True,jac=False)[0].detach()
            #loss2 = ((ert_fake-ert_sample)**2).mean()
            #loss2_history.append(loss2.item())

            rand_ert = torch.randn(BATCH_SIZE,OUTPUT_SIZE,device=device)
            ert_fake_code = Inverse_flow(rand_ert,obs_sample,rev=True,jac=False)[0].detach()
            ert_fake = Mod_Normal_Flow(ert_fake_code,rev=True,jac=False)[0].detach()

            X,Y = ert_sample.cpu().numpy(),ert_fake.cpu().numpy()
            fig, axes = plt.subplots(2,2, figsize=(12, 12), dpi=100, sharex=True, sharey=True)
            N1 = model_pair['Lm1_den']
            N2 = model_pair['Lm1_vpv']
            #N1 = 12
            #N2 = 82
            axes[0,0].scatter(X[:, N1], X[:, N2],edgecolors='black')
            axes[0,1].scatter(Y[:, N1], Y[:, N2],edgecolors='black')

            ert_fake_code = Inverse_flow(rand_ert,obs_tensor,rev=True,jac=False)[0].detach()
            ert_fake = Mod_Normal_Flow(ert_fake_code,rev=True,jac=False)[0].detach()
            ert_fake = ert_fake.cpu().numpy()
            axes[1,1].scatter(ert_fake[:, N1], ert_fake[:, N2],edgecolors='black')

            fig.tight_layout()
            torch.cuda.empty_cache(); gc.collect()
            plt.show(fig); plt.close(fig)

            #画w2
            fig, ax = plt.subplots(1, 1, figsize=(12, 3), dpi=100)
            ax.set_title('loss', fontsize=12)
            ax.plot(ewma(train_loss_history), c='red', label='train_loss')
            ax.plot(ewma(test_loss_history), c='blue', label='test_loss')
            #ax.plot(ewma(loss2_history),c='green',label='loss2')
            fig.tight_layout()
            plt.show(fig); plt.close(fig) 

def train_inverse_flow_original1(Obs_Normal_Flow,Inverse_flow,opt_inv,train_data,var_np,obs_tensor,test_data,label=True,epochs=10001):
    train_loss_history = []
    test_loss_history = []
    #loss2_history = []
    Obs_Normal_Flow.eval();
    err_dist = torch.distributions.MultivariateCalculate the special weighting factor ww.Normal(torch.zeros(INPUT_SIZE,dtype=torch.float32),
                                                                    torch.diag(torch.from_numpy(np.power(var_np,2))).
                                                                    type(torch.float32))
    for it in tqdm(range(epochs)):
        error = err_dist.sample((BATCH_SIZE,)).cuda()
        if label == True:
            obs_sample,ert_sample, weight = next(iter(train_data))
            obs_sample, ert_sample, weight = obs_sample.to(device), ert_sample.to(device),weight.to(device)
        else:
            obs_sample,ert_sample = next(iter(train_data))
            obs_sample, ert_sample = obs_sample.to(device), ert_sample.to(device)
            
        obs_code = Obs_Normal_Flow(obs_sample+error)[0].detach()
    
        Inverse_flow.train()
        opt_inv.zero_grad()
        z,log_jac_det = Inverse_flow(ert_sample,obs_code)
        if label == True:
            #loss = (0.5*torch.sum(z**2,1)-log_jac_det)*weight
            loss = (0.5*torch.sum(z**2,1)-log_jac_det)
        else:
            loss = (0.5*torch.sum(z**2,1)-log_jac_det)
        loss = loss.mean()/OUTPUT_SIZE

        rand_ert = torch.randn(BATCH_SIZE,OUTPUT_SIZE,device=device)
        ert_fake = Inverse_flow(rand_ert,obs_code,rev=True,jac=False)[0].detach()
        loss2 = ((ert_fake-ert_sample)**2).mean()
        loss = loss+loss2*10.

        loss.backward()
        opt_inv.step()
        opt_inv.zero_grad()
        if it%100 == 0:
            #train_loss_history.append(loss.item())
            train_loss_history.append((loss-loss2*10.).item())
            Inverse_flow.eval()
            clear_output(wait=True)
            error = err_dist.sample((BATCH_SIZE,)).cuda()
            
            if label == True:
                obs_sample,ert_sample,_ = next(iter(test_data))
            else:
                obs_sample,ert_sample = next(iter(test_data))
            obs_sample, ert_sample = obs_sample.to(device), ert_sample.to(device)
            obs_code = Obs_Normal_Flow(obs_sample+error)[0].detach()
            
            z,log_jac_det = Inverse_flow(ert_sample,obs_code)
            if label == True:
                #loss = (0.5*torch.sum(z**2,1)-log_jac_det)*weight
                loss = (0.5*torch.sum(z**2,1)-log_jac_det)
            else:
                loss = (0.5*torch.sum(z**2,1)-log_jac_det)
            loss = loss.mean()/OUTPUT_SIZE
            test_loss_history.append(loss.item())

            #rand_ert = torch.randn(BATCH_SIZE,OUTPUT_SIZE,device=device)
            #ert_fake_code = Inverse_flow(rand_ert,obs_code,rev=True,jac=False)[0].detach()
            #ert_fake = Mod_Normal_Flow(ert_fake_code,rev=True,jac=False)[0].detach()
            #loss2 = ((ert_fake-ert_sample)**2).mean()
            #loss2_history.append(loss2.item())

            rand_ert = torch.randn(BATCH_SIZE,OUTPUT_SIZE,device=device)
            ert_fake = Inverse_flow(rand_ert,obs_code,rev=True,jac=False)[0].detach()
            
            X,Y = ert_sample.cpu().numpy(),ert_fake.cpu().numpy()
            fig, axes = plt.subplots(2,2, figsize=(12, 12), dpi=100, sharex=True, sharey=True)
            N1 = model_pair['Lm1_den']
            N2 = model_pair['Lm1_vpv']
            #N1 = 12
            #N2 = 82
            axes[0,0].scatter(X[:, N1], X[:, N2],edgecolors='black')
            axes[0,1].scatter(Y[:, N1], Y[:, N2],edgecolors='black')

            obs_code = Obs_Normal_Flow(obs_tensor)[0].detach()
            ert_fake = Inverse_flow(rand_ert,obs_code,rev=True,jac=False)[0].detach().cpu().numpy()
            axes[1,1].scatter(ert_fake[:, N1], ert_fake[:, N2],edgecolors='black')

            fig.tight_layout()
            torch.cuda.empty_cache(); gc.collect()
            plt.show(fig); plt.close(fig)

            #画w2
            fig, ax = plt.subplots(1, 1, figsize=(12, 3), dpi=100)
            ax.set_title('loss', fontsize=12)
            ax.plot(ewma(train_loss_history), c='red', label='train_loss')
            ax.plot(ewma(test_loss_history), c='blue', label='test_loss')
            #ax.plot(ewma(loss2_history),c='green',label='loss2')
            fig.tight_layout()
            plt.show(fig); plt.close(fig) 


def train_inverse_flow_original(Inverse_flow,opt_inv,train_data,var_np,obs_tensor,test_data,label=False,epochs=10001):
    #BATCH_SIZE = 256
    obs_tensor = obs_tensor[:BATCH_SIZE]
    
    train_loss_history = []
    test_loss_history = []
    #loss2_history = []
    #Obs_Normal_Flow.eval();
    err_dist = torch.distributions.MultivariateNormal(torch.zeros(INPUT_SIZE,dtype=torch.float32),
                                                                    torch.diag(torch.from_numpy(np.power(var_np,2))).
                                                                    type(torch.float32))
    for it in tqdm(range(epochs)):
        error = err_dist.sample((BATCH_SIZE,)).cuda()
        if label:
            obs_sample, ert_sample,_ = next(iter(train_data))
        else:
            obs_sample, ert_sample = next(iter(train_data))
        obs_sample, ert_sample = obs_sample.to(device)+error, ert_sample.to(device)
        #obs_sample, ert_sample = obs_sample.to(device), ert_sample.to(device)
        
        Inverse_flow.train()
        opt_inv.zero_grad()
        z,log_jac_det = Inverse_flow(ert_sample,obs_sample)
        loss = 0.5*torch.sum(z**2,1)-log_jac_det
        loss = loss.mean()/OUTPUT_SIZE

        #rand_ert = torch.randn(BATCH_SIZE,OUTPUT_SIZE,device=device)
        #ert_fake = Inverse_flow(rand_ert,obs_sample,rev=True,jac=False)[0].detach()
        #loss2 = ((ert_fake-ert_sample)**2).mean()
        #loss = loss+loss2*10.
        
        loss.backward()
        opt_inv.step()
        opt_inv.zero_grad()
        if it%100 == 0:
            #train_loss_history.append((loss-loss2*10).item())
            train_loss_history.append(loss.item())
            Inverse_flow.eval()
            clear_output(wait=True)
            error = err_dist.sample((BATCH_SIZE,)).cuda()
            if label:
                obs_sample,ert_sample,_ = next(iter(test_data))
            else:
                obs_sample,ert_sample = next(iter(test_data))
            obs_sample, ert_sample = obs_sample.to(device)+error, ert_sample.to(device)
            #obs_sample, ert_sample = obs_sample.to(device), ert_sample.to(device)
            z,log_jac_det = Inverse_flow(ert_sample,obs_sample)
            loss = 0.5*torch.sum(z**2,1)-log_jac_det
            loss = loss.mean()/OUTPUT_SIZE
            test_loss_history.append(loss.item())

            #rand_ert = torch.randn(BATCH_SIZE,OUTPUT_SIZE,device=device)
            #ert_fake = Inverse_flow(rand_ert,obs_sample,rev=True,jac=False)[0].detach()
            #loss2 = ((ert_fake-ert_sample)**2).mean()
            #loss2_history.append(loss2.item())
           
            rand_ert = torch.randn(BATCH_SIZE,OUTPUT_SIZE,device=device)
            ert_fake = Inverse_flow(rand_ert,obs_sample,rev=True,jac=False)[0].detach()
            
            X,Y = ert_sample.cpu().numpy(),ert_fake.cpu().numpy()
            fig, axes = plt.subplots(2,2, figsize=(12, 12), dpi=100, sharex=True, sharey=True)
            N1 = model_pair['Lm1_den']
            N2 = model_pair['Lm1_vpv']
            axes[0,0].scatter(X[:, N1], X[:, N2],edgecolors='black')
            axes[0,1].scatter(Y[:, N1], Y[:, N2],edgecolors='black')

            #obs_code = Obs_Normal_Flow(obs_tensor)[0].detach()
            ert_fake,log_jac_fake = Inverse_flow(rand_ert,obs_tensor,rev=True,jac=False)
            ert_fake = ert_fake.detach().cpu().numpy()  
            
            axes[1,1].scatter(ert_fake[:, N1], ert_fake[:, N2],edgecolors='black')

            fig.tight_layout()
            torch.cuda.empty_cache(); gc.collect()
            plt.show(fig); plt.close(fig)

            #画w2
            fig, ax = plt.subplots(1, 1, figsize=(12, 3), dpi=100)
            ax.set_title('loss', fontsize=12)
            ax.plot(ewma(train_loss_history), c='red', label='train_loss')
            ax.plot(ewma(test_loss_history), c='blue', label='test_loss')
            #ax.plot(ewma(loss2_history), c='green', label='loss2')
            fig.tight_layout()
            plt.show(fig); plt.close(fig) 


def train_mod_normalization_zero(Mod_Normal_Flow,opt_mod,train_data,var_np,epochs=10001):
    loss_history = []
    for it in tqdm(range(epochs)):
        Mod_Normal_Flow.train(True)
        obs_sample,ert_sample = next(iter(train_data))
        ert_sample = ert_sample.to(device)
        opt_mod.zero_grad()
        rand_ert = torch.randn(BATCH_SIZE,OUTPUT_SIZE,device=device)*0.01
        z,log_jac_det = Mod_Normal_Flow(ert_sample+rand_ert)
        loss = 0.5*torch.sum(z**2,1)-log_jac_det
        loss = loss.mean()/OUTPUT_SIZE
        loss_history.append(loss.item())
        loss.backward()
        opt_mod.step()
        opt_mod.zero_grad()
        if it%100 == 0:
            Mod_Normal_Flow.eval()
            clear_output(wait=True)
            N1 = model_pair['Lm1_den']
            N2 = model_pair['Lm1_vpv']
            obs_sample,ert_sample = next(iter(train_data))
            ert_sample = ert_sample.to(device)
            rand_data = torch.randn(BATCH_SIZE,OUTPUT_SIZE,device=device)
            ert_fake,_ = Mod_Normal_Flow(rand_data,rev=True,jac=False)
            X,Y = ert_sample.detach().cpu().numpy(), ert_fake.detach().cpu().numpy()

            fig, axes = plt.subplots(1,2, figsize=(12, 6), dpi=100, sharex=True, sharey=True)
            axes[0].scatter(X[:,N1],X[:,N2],edgecolors='black')
            axes[1].scatter(Y[:,N1],Y[:,N2],edgecolors='black')

            fig.tight_layout()
            torch.cuda.empty_cache(); gc.collect()
            plt.show(fig); plt.close(fig)

            fig, ax = plt.subplots(1, 1, figsize=(12, 3), dpi=100)
            ax.set_title('loss', fontsize=12)
            ax.plot(ewma(loss_history), c='blue', label='Estimated Cost')
            fig.tight_layout()
            plt.show(fig); plt.close(fig)