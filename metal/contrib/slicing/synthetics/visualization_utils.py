import matplotlib.pyplot as plt
import numpy as np

def visualize_data(X, Y, C, L):
    # show data by class
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 2, 1)
    plt.title('Data by Classes')
    plt.scatter(X[Y==1,0], X[Y==1,1], label="$y=+1$", c='C1')
    plt.scatter(X[Y==-1,0], X[Y==-1,1], label="$y=-1$", c='C0')
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.legend()

    # show data by slice
    plt.subplot(2, 2, 2)
    plt.title('Data by Slice')
    plt.scatter(X[C==0,0], X[C==0,1], label="$S_0$", c='C0')
    plt.scatter(X[C==1,0], X[C==1,1], label="$S_1$", c='C1')
    plt.scatter(X[C==2,0], X[C==2,1], label="$S_2$", c='C2')
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.legend()

    # LFs targeting slices
    plt.subplot(2, 2, 3)
    plt.title('LFs ($\lambda_i$) Targeting Slices ($S_i$)')
    plt.scatter(X[L[:,0]!=0,0], X[L[:,0]!=0,1], label="$\lambda_0$", c='C0')
    plt.scatter(X[L[:,1]!=0,0], X[L[:,1]!=0,1], label="$\lambda_1$", c='C1')
    plt.scatter(X[L[:,2]!=0,0], X[L[:,2]!=0,1], label="$\lambda_2$", c='C2')
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.title('$\lambda_2$ votes')
    # first plot underlying slice
    plt.scatter(X[C==2,0], X[C==2,1], label="$S_2$", s=0.1, c='red')
    plt.scatter(X[L[:,2]==1,0], X[L[:,2]==1,1], label="$\lambda_2=+1$", s=0.2, c='C1')
    plt.scatter(X[L[:,2]==-1,0], X[L[:,2]==-1,1], label="$\lambda_2=-1$", s=0.2, c='C0')    
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.legend()
    plt.show()

def plot_slice_scores(results, slice_name="S2", xlabel="Overlap Proportion"):
    for model_name, model_scores in results.items(): 
        x_range = model_scores.keys()
    
        # take average value across trials
        scores_collected = [
            np.mean(np.array([s[slice_name] for s in model_scores[x]]))
            for x in x_range
        ]

        plt.plot(x_range, scores_collected, label=model_name)
    
    # print x-axis in precision 2
    x_range = ["%.2f" % float(x) for x in x_range]

    plt.title(f"Accuracy on {slice_name} vs. {xlabel}")
    plt.xlabel(xlabel)
    plt.ylabel(f"Accuracy on {slice_name}")
    plt.ylim(bottom=0, top=1)
    plt.legend()
