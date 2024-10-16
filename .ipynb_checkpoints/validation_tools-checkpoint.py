import torch
from tools import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from IPython.display import clear_output

OUTPUT_SIZE = 152
INPUT_SIZE = 152
BATCH_SIZE = 256

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
            N1 = 10
            N2 = 21
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
            obs_sample,ert_sample,_ = next(iter(train_data))
        else:
            obs_sample,ert_sample = next(iter(train_data))
        ert_sample = ert_sample.to(device)
        opt_mod.zero_grad()
        z,log_jac_det = Mod_Normal_Flow(ert_sample)
        loss = 0.5*torch.sum(z**2,1)-log_jac_det
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
                obs_sample,ert_sample,_ = next(iter(test_data))
            else:
                obs_sample,ert_sample = next(iter(test_data))
            ert_sample = ert_sample.to(device)
            z,log_jac_det = Mod_Normal_Flow(ert_sample)
            loss = 0.5*torch.sum(z**2,1)-log_jac_det
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

def train_inverse_flow_validation(Obs_Normal_Flow,Mod_Normal_Flow,Inverse_flow,opt_inv,train_data,var_np,obs_tensor,test_data,test_ert,label=True,epochs=10001):
    train_loss_history = []
    test_loss_history = []
    cross_entropy_history1 = []
    cross_entropy_history2 = []
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
            loss = (0.5*torch.sum(z**2,1)-log_jac_det)*weight
            #loss = (0.5*torch.sum(z**2,1)-log_jac_det)
        else:
            loss = (0.5*torch.sum(z**2,1)-log_jac_det)
        loss = loss.mean()/OUTPUT_SIZE
        loss.backward()
        opt_inv.step()
        opt_inv.zero_grad()
        if it%100 == 0:
            train_loss_history.append(loss.item())
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
                loss = (0.5*torch.sum(z**2,1)-log_jac_det)*weight
                #loss = (0.5*torch.sum(z**2,1)-log_jac_det)
            else:
                loss = (0.5*torch.sum(z**2,1)-log_jac_det)
            loss = loss.mean()/OUTPUT_SIZE
            test_loss_history.append(loss.item())

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

            obs_code = Obs_Normal_Flow(obs_tensor.to(device))[0].detach()
            ert_fake_code = Inverse_flow(rand_ert,obs_code,rev=True,jac=False)[0].detach()
            ert_fake = Mod_Normal_Flow(ert_fake_code,rev=True,jac=False)[0].detach()
            ert_fake = ert_fake.cpu().numpy()

            ert_test = test_ert.to(device)
            ert_test_code,log_jac_det1 = Mod_Normal_Flow(ert_test)
            #cross_entropy1 = (0.5*torch.sum(ert_test_code**2,1)-log_jac_det1)
            cross_entropy1 = -log_jac_det1
            cross_entropy1 = cross_entropy1.mean()/OUTPUT_SIZE
            cross_entropy_history1.append(cross_entropy1.item())
            
            z,log_jac_det = Inverse_flow(ert_test_code,obs_code)
            cross_entropy2 = (0.5*torch.sum(z**2,1)-log_jac_det)
            cross_entropy2 = cross_entropy2.mean()/OUTPUT_SIZE
            cross_entropy_history2.append(cross_entropy2.item())
            axes[1,0].scatter(test_ert.detach().cpu().numpy()[:, N1], test_ert.detach().cpu().numpy()[:, N2],edgecolors='black')
            axes[1,1].scatter(ert_fake[:, N1], ert_fake[:, N2],edgecolors='black')

            fig.tight_layout()
            torch.cuda.empty_cache(); gc.collect()
            plt.show(fig); plt.close(fig)

            #画w2
            fig, ax = plt.subplots(1, 1, figsize=(12, 3), dpi=100)
            ax.set_title('loss', fontsize=12)
            ax.plot(ewma(train_loss_history), c='red', label='train_loss')
            ax.plot(ewma(test_loss_history), c='blue', label='test_loss')
            ax.plot(ewma(cross_entropy_history2),c='green',label='cross_entropy')
            fig.tight_layout()
            plt.show(fig); plt.close(fig)
    return cross_entropy_history1,cross_entropy_history2


def train_inverse_flow_original2_validation(Inverse_flow,opt_inv,train_data,var_np,obs_fix,test_data,test_ert,epochs=10001):
    train_loss_history = []
    test_loss_history = []
    cross_entropy_history1 = []
    cross_entropy_history2 = []
    save_num = 0
    err_dist = torch.distributions.MultivariateNormal(torch.zeros(INPUT_SIZE,dtype=torch.float32),
                                                                    torch.diag(torch.from_numpy(np.power(var_np,2))).
                                                                    type(torch.float32))
    for it in tqdm(range(epochs)):
        error = err_dist.sample((BATCH_SIZE,)).cuda()
        obs_sample,ert_sample = next(iter(train_data))
        obs_sample, ert_sample = obs_sample.to(device)+error, ert_sample.to(device)
        
        Inverse_flow.train()
        opt_inv.zero_grad()
        z,log_jac_det = Inverse_flow(ert_sample,obs_sample)
        loss = 0.5*torch.sum(z**2,1)-log_jac_det
        loss = loss.mean()/OUTPUT_SIZE
        loss.backward()
        opt_inv.step()
        opt_inv.zero_grad()
        if it%100 == 0:
            train_loss_history.append(loss.item())
            Inverse_flow.eval()
            clear_output(wait=True)
            error = err_dist.sample((BATCH_SIZE,)).cuda()
            obs_sample, ert_sample = next(iter(test_data))
            obs_sample, ert_sample = obs_sample.to(device), ert_sample.to(device)
            
            z,log_jac_det = Inverse_flow(ert_sample,obs_sample)
            loss = 0.5*torch.sum(z**2,1)-log_jac_det
            loss = loss.mean()/OUTPUT_SIZE
            test_loss_history.append(loss.item())
           
            rand_ert = torch.randn(BATCH_SIZE,OUTPUT_SIZE,device=device)
            ert_fake = Inverse_flow(rand_ert,obs_sample,rev=True,jac=False)[0].detach()
            
            X,Y = ert_sample.cpu().numpy(),ert_fake.cpu().numpy()
            fig, axes = plt.subplots(2,2, figsize=(12, 12), dpi=100, sharex=True, sharey=True)
            N1 = model_pair['Lm1_den']
            N2 = model_pair['Lm1_vpv']
            axes[0,0].scatter(X[:, N1], X[:, N2],edgecolors='black')
            axes[0,1].scatter(Y[:, N1], Y[:, N2],edgecolors='black')

            ert_fake,log_jac_fake = Inverse_flow(rand_ert,obs_fix.to(device),rev=True,jac=False)
            cross_entropy2 = (0.5*torch.sum(rand_ert**2,1)+log_jac_fake)
            cross_entropy2 = cross_entropy2.mean()/OUTPUT_SIZE
            ert_fake = ert_fake.detach().cpu().numpy()
            cross_entropy_history2.append((cross_entropy2).item())

            ert_test = test_ert.to(device)
            z,log_jac_det = Inverse_flow(ert_test,obs_fix.to(device))
            cross_entropy = (0.5*torch.sum(z**2,1)-log_jac_det)
            cross_entropy = cross_entropy.mean()/OUTPUT_SIZE
            
            cross_entropy_history1.append((cross_entropy).item())

            axes[1,0].scatter(test_ert[:, N1], test_ert[:, N2],edgecolors='black')
            axes[1,1].scatter(ert_fake[:, N1], ert_fake[:, N2],edgecolors='black')

            fig.tight_layout()
            torch.cuda.empty_cache(); gc.collect()
            plt.show(fig); plt.close(fig)

            #画w2
            fig, ax = plt.subplots(1, 1, figsize=(12, 3), dpi=100)
            ax.set_title('loss', fontsize=12)
            ax.plot(ewma(train_loss_history), c='red', label='train_loss')
            ax.plot(ewma(test_loss_history), c='blue', label='test_loss')
            ax.plot(ewma(cross_entropy_history1),c='green',label='cross_entropy')
            fig.tight_layout()
            plt.show(fig); plt.close(fig)
    
        if it%10000 == 0:
            file_name = './net/Inverse_flow0_val_{0}.pth'.format(save_num)
            torch.save(Inverse_flow.state_dict(),file_name)
            save_num = save_num + 1
    return cross_entropy_history1,cross_entropy_history2

def train_inverse_flow_original_validation(Obs_Normal_Flow,Inverse_flow,opt_inv,train_data,var_np,obs_tensor,test_data,test_ert,epochs=10001):
    train_loss_history = []
    test_loss_history = []
    cross_entropy_history = []
    save_num = 0
    Obs_Normal_Flow.eval();
    err_dist = torch.distributions.MultivariateNormal(torch.zeros(INPUT_SIZE,dtype=torch.float32),
                                                                    torch.diag(torch.from_numpy(np.power(var_np,2))).
                                                                    type(torch.float32))
    for it in tqdm(range(epochs)):
        error = err_dist.sample((BATCH_SIZE,)).cuda()
        obs_sample,ert_sample = next(iter(train_data))
        obs_sample, ert_sample = obs_sample.to(device)+error, ert_sample.to(device)
        obs_code = Obs_Normal_Flow(obs_sample+error)[0].detach()
        
        Inverse_flow.train()
        opt_inv.zero_grad()
        z,log_jac_det = Inverse_flow(ert_sample,obs_code)
        loss = 0.5*torch.sum(z**2,1)-log_jac_det
        loss = loss.mean()/OUTPUT_SIZE
        loss.backward()
        opt_inv.step()
        opt_inv.zero_grad()
        if it%100 == 0:
            train_loss_history.append(loss.item())
            Inverse_flow.eval()
            clear_output(wait=True)
            error = err_dist.sample((BATCH_SIZE,)).cuda()
            obs_sample,ert_sample = next(iter(test_data))
            obs_sample, ert_sample = obs_sample.to(device), ert_sample.to(device)
            obs_code = Obs_Normal_Flow(obs_sample+error)[0].detach()
            z,log_jac_det = Inverse_flow(ert_sample,obs_code)
            loss = 0.5*torch.sum(z**2,1)-log_jac_det
            loss = loss.mean()/OUTPUT_SIZE
            test_loss_history.append(loss.item())
           
            rand_ert = torch.randn(BATCH_SIZE,OUTPUT_SIZE,device=device)
            ert_fake = Inverse_flow(rand_ert,obs_code,rev=True,jac=False)[0].detach()
            
            X,Y = ert_sample.cpu().numpy(),ert_fake.cpu().numpy()
            fig, axes = plt.subplots(2,2, figsize=(12, 12), dpi=100, sharex=True, sharey=True)
            N1 = model_pair['Lm1_den']
            N2 = model_pair['Lm1_vpv']
            axes[0,0].scatter(X[:, N1], X[:, N2],edgecolors='black')
            axes[0,1].scatter(Y[:, N1], Y[:, N2],edgecolors='black')

            obs_code = Obs_Normal_Flow(obs_tensor)[0].detach()
            ert_fake,log_jac_fake = Inverse_flow(rand_ert,obs_code,rev=True,jac=False)
            ert_fake = ert_fake.detach().cpu().numpy()

            ert_test = test_ert.to(device)
            z,log_jac_det = Inverse_flow(ert_test,obs_code)
            cross_entropy = (0.5*torch.sum(z**2,1)-log_jac_det)
            cross_entropy = cross_entropy.mean()/OUTPUT_SIZE
            cross_entropy_history.append(cross_entropy.item())
            
            axes[1,0].scatter(test_ert[:, N1], test_ert[:, N2],edgecolors='black')
            axes[1,1].scatter(ert_fake[:, N1], ert_fake[:, N2],edgecolors='black')

            fig.tight_layout()
            torch.cuda.empty_cache(); gc.collect()
            plt.show(fig); plt.close(fig)

            #画w2
            fig, ax = plt.subplots(1, 1, figsize=(12, 3), dpi=100)
            ax.set_title('loss', fontsize=12)
            ax.plot(ewma(train_loss_history), c='red', label='train_loss')
            ax.plot(ewma(test_loss_history), c='blue', label='test_loss')
            fig.tight_layout()
            plt.show(fig); plt.close(fig)

    return cross_entropy_history
