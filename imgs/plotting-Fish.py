
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.rcParams.update({
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 13
})
epochs = np.array([2, 4, 8, 10, 14, 30])

models = ['Baseline', 'Closest20', '10GAN', '100GAN']

#################FISH#########################
#####GROUPNET#################
# Data for Baseline
base_dcs = np.array([0.5, 0.51, 0.39, 0.23, 0.22, 0.1])
base_dcs_std = np.array([0.05, 0.05, 0.06, 0.06, 0.06, 0.04])

base_dus =np.array([0.54 ,0.58, 0.67, 0.63, 0.75, 0.91])
base_dus_std = np.array([0.02, 0.03, 0.05, 0.06, 0.05, 0.03])

base_recall =np.array([0.64, 0.60, 0.96, 1, 1, 1])
base_recall_std = np.array([0.22, 0.26, 0.06, 0, 0, 0])

base_acc = np.array([0.51, 0.5, 0.68, 0.69, 0.7, 0.7])
base_acc_std =  np.array([0.24, 0.27, 0.16, 0.15, 0.15, 0.15])

base_per =  np.array([0.52, 0.51, 0.64, 0.64, 0.65, 0.65])
base_per_std =  np.array([0.21, 0.23, 0.14, 0.13, 0.13, 0.13])

base_f1 = np.array([0.57, 0.55, 0.76, 0.77, 0.78, 0.78])
base_f1_std =  np.array([0.21, 0.24, 0.1, 0.09, 0.09, 0.09])





# Data for Closest
closest_dus = np.array([0.73, 0.7, 0.73, 0.85, 0.92, 0.97])
closest_dus_std = np.array([0.01, 0.04, 0.03,0.12, 0.02, 0.02])

closest_dcs = np.array([0.72, 0.67, 0.46, 0.23, 0.16, 0.04])
closest_dcs_std = np.array([0.01, 0.02, 0.09, 0.03, 0.1, 0.06])

closest_recall = np.array([0.4, 0.4, 0.77, 0.98, 0.99, 1])
closest_recall_std = np.array([0.3, 0.3, 0.17, 0.04, 0.02, 0])

closest_acc = np.array([0.4, 0.4, 0.58, 0.69, 0.69, 0.7])
closest_acc_std =  np.array([0.3, 0.3, 0.22, 0.15, 0.15, 0.15])

closest_per =  np.array([0.4, 0.4, 0.58, 0.64, 0.64, 0.65])
closest_per_std =  np.array([0.3, 0.3, 0.18, 0.13, 0.13, 0.13])

closest_f1 = np.array([0.4 , 0.4, 0.66, 0.77, 0.77, 0.78])
closest_f1_std =  np.array([0.3, 0.3, 0.17, 0.1, 0.09, 0.09])




# Data for GAN100
gan100_dcs = np.array([0.48, 0.49, 0.47, 0.49, 0.5, 0.34])
gan100_dcs_std = np.array([0.0,0.01 , 0.01, 0.01, 0.04, 0.05])

gan100_dus = np.array([0.51, 0.5, 0.5, 0.51, 0.51, 0.69])
gan100_dus_std = np.array([0, 0, 0, 0.01, 0.04, 0.05])

gan100_recall = np.array([0.96, 0.96, 0.75, 0.82, 0.66, 0.99])
gan100_recall_std = np.array([0.07, 0.06, 0.19, 0.13, 0.2, 0.01])

gan100_acc = np.array([0.61, 0.62, 0.56, 0.58, 0.52, 0.69])
gan100_acc_std =  np.array([0.13, 0.14, 0.22, 0.18, 0.23, 0.15])

gan100_per =  np.array([0.57, 0.59, 0.56, 0.57, 0.53, 0.65])
gan100_per_std =  np.array([0.1, 0.11, 0.18, 0.14, 0.2, 0.13])

gan100_f1 = np.array([0.71, 0.72, 0.64, 0.66, 0.58, 0.77])
gan100_f1_std =  np.array([0.08, 0.09, 0.18, 0.13, 0.2, 0.09])


# Data for GAN10
gan10_dcs =np.array([0.49, 0.49, 0.49, 0.49, 0.48, 0.16])
gan10_dcs_std = np.array([0.01, 0.0, 0.0, 0.0, 0.01, 0.04])

gan10_dus = np.array([0.51, 0.5, 0.5, 0.5, 0.51, 0.75])
gan10_dus_std = np.array([0.01, 0.0, 0.0,0.0, 0.01, 0.03])

gan10_recall = np.array([0.88, 0.71, 0.85, 0.89, 0.87, 1])
gan10_recall_std = np.array([0.12, 0.21, 0.11, 0.13, 0.12, 0.0])
gan10_acc = np.array([0.61, 0.53, 0.61, 0.59, 0.62, 0.7])
gan10_acc_std =  np.array([0.18, 0.22, 0.17, 0.17, 0.18, 0.15])

gan10_per =  np.array([0.59, 0.53, 0.58, 0.57, 0.6, 0.65])
gan10_per_std =  np.array([0.14, 0.19, 0.13, 0.13, 0.14, 0.13])

gan10_f1 = np.array([0.7, 0.6, 0.7, 0.69, 0.71, 0.78])
gan10_f1_std =  np.array([0.12, 0.19, 0.12 , 0.12, 0.12, 0.09])



# Data for 10GM-no mission
gm10_nomission = np.array([0.5, 0.51, 0.5, 0.5, 0.5, 0.20])
gm10_nomission_std = np.array([0.0, 0.01, 0.0, 0.0, 0.01, 0.05])

# Data for 100GM-no mission
gm100_nomission = np.array([0.49, 0.5, 0.5, 0.5, 0.5 , 0.28])
gm100_nomission_std = np.array([0.0, 0, 0.00, 0.0, 0.04,  0.04])



gan250_dcs =np.array([0.49, 0.5, 0.51, 0.5, 0.49, 0.49])
gan250_dcs_std = np.array([0, 0,0, 0, 0.01, 0.01])

gan250_dus = np.array([0.5, 0.51, 0.51, 0.51, 0.5, 0.51])
gan250_dus_std = np.array([0.0, 0.0, 0.0,0.0, 0.01, 0.01])

gan250_recall = np.array([0.72, 0.47, 0.42, 0.53,0.69 , 0.67])
gan250_recall_std = np.array([0.2, 0.28, 0.3, 0.25, 0.23, 0.22])

gan250_acc = np.array([0.49, 0.42, 0.4, 0.45, 0.52, 0.51])
gan250_acc_std =  np.array([0.2, 0.28, 0.29, 0.26, 0.23, 0.24])

gan250_per = np.array([0.5, 0.43, 0.41, 0.47, 0.52, 0.52])
gan250_per_std =  np.array([0.15, 0.26, 0.29, 0.24, 0.19, 0.2])

gan250_f1 = np.array([0.59, 0.45, 0.41, 0.49, 0.59, 0.58])
gan250_f1_std =  np.array([0.17, 0.27, 0.29 , 0.24, 0.2, 0.2])

gan250_nomission =np.array([0.5, 0.5, 0.51, 0.5, 0.5, 0.5])
gan250_nomission_std = np.array([0.0, 0, 0.00, 0.0, 0.01,  0.01])


fig, axs = plt.subplots(1, 3, figsize=(20, 5))

axs[0].plot(epochs, base_dcs, label='Base-Rigid')
axs[0].fill_between(epochs, base_dcs - base_dcs_std, base_dcs + base_dcs_std, alpha=0.2)
axs[0].plot(epochs, closest_dcs, label='Base-Smooth')
axs[0].fill_between(epochs, closest_dcs - closest_dcs_std, closest_dcs + closest_dcs_std, alpha=0.2)
axs[0].plot(epochs, gan10_dcs, label='10GAN' ,color='green')
axs[0].fill_between(epochs, gan10_dcs - gan10_dcs_std, gan10_dcs + gan10_dcs_std, alpha=0.2,color='green')
axs[0].plot(epochs, gan100_dcs, label='100GAN',color='red')
axs[0].fill_between(epochs, gan100_dcs - gan100_dcs_std, gan100_dcs + gan100_dcs_std, alpha=0.2,color='red')
axs[0].plot(epochs, gan250_dcs, label='250GAN',color='purple')
axs[0].fill_between(epochs, gan250_dcs - gan250_dcs_std, gan250_dcs + gan250_dcs_std, alpha=0.2,color='purple')
axs[0].set_title("(a) Discriminator's controlled Agents' Score\nFish dataset - GroupNet Scheme", wrap=True)
axs[0].set_xlabel("Discriminator's epoch")
axs[0].set_ylabel("Score")
axs[0].legend(loc='lower left')
axs[0].grid(False)

# axs[1].plot(epochs, base_dus, label='Base-Rigid')
# axs[1].fill_between(epochs, base_dus - base_dus_std, base_dus + base_dus_std, alpha=0.2)
# axs[1].plot(epochs, closest_dus, label='Base-Smooth')
# axs[1].fill_between(epochs, closest_dus - closest_dus_std, closest_dus + closest_dus_std, alpha=0.2)
# axs[1].plot(epochs, gan10_dus, label='10GAN',color='green')
# axs[1].fill_between(epochs, gan10_dus - gan10_dus_std, gan10_dus + gan10_dus_std, alpha=0.2,color='green')
# axs[1].plot(epochs, gan100_dus, label='100GAN',color='red')
# axs[1].fill_between(epochs, gan100_dus - gan100_dus_std, gan100_dus + gan100_dus_std, alpha=0.2,color='red')
# axs[1].plot(epochs, gan250_dus, label='250GAN',color='purple')
# axs[1].fill_between(epochs, gan250_dus - gan250_dus_std, gan250_dus + gan250_dus_std, alpha=0.2,color='purple')
# axs[1].set_title("Discriminator's uncntrolled Agents' Scores (DUS)\nFish dataset - GroupNet Scheme", wrap=True)
# axs[1].set_xlabel("Discriminator's epoch")
# axs[1].set_ylabel("Score")
# axs[1].legend()
# axs[1].grid(False)

axs[1].plot(epochs, gm10_nomission, label='10GM-no mission',  color='green')
axs[1].fill_between(epochs, gm10_nomission - gm10_nomission_std, gm10_nomission + gm10_nomission_std, alpha=0.2,  color='green')

axs[1].plot(epochs, gm100_nomission, label='100GM-no mission',  color='red')
axs[1].fill_between(epochs, gm100_nomission - gm100_nomission_std, gm100_nomission + gm100_nomission_std, alpha=0.2,  color='red')

axs[1].plot(epochs, gan250_nomission, label='250GM-no mission',  color='purple')
axs[1].fill_between(epochs, gan250_nomission - gan250_nomission_std, gan250_nomission +gan250_nomission_std, alpha=0.2,  color='purple')

axs[1].set_title("(b) Controlled (No Mission) Agents' Scores\nFish dataset - GroupNet Scheme")
axs[1].set_xlabel("Discriminator's epoch")
axs[1].set_ylabel("Score")
axs[1].legend()
axs[1].grid(False)

axs[2].plot(epochs, base_recall, label='Base-Rigid')
axs[2].fill_between(epochs, base_recall - base_recall_std, base_recall + base_recall_std, alpha=0.2)
axs[2].plot(epochs, closest_recall, label='Base-Smooth')
axs[2].fill_between(epochs, closest_recall - closest_recall_std, closest_recall + closest_recall_std, alpha=0.2)
axs[2].plot(epochs, gan10_recall, label='10GAN',color='green')
axs[2].fill_between(epochs, gan10_recall - gan10_recall_std, gan10_recall + gan10_recall_std, alpha=0.2,color='green')
axs[2].plot(epochs, gan100_recall, label='100GAN',color='red')
axs[2].fill_between(epochs, gan100_recall - gan100_recall_std, gan100_recall +gan100_recall_std, alpha=0.2,color='red')
axs[2].plot(epochs, gan250_recall, label='250GAN',color='purple')
axs[2].fill_between(epochs, gan250_recall - gan250_recall_std, gan250_recall + gan250_recall_std, alpha=0.2,color='purple')

axs[2].set_title("(c) Agents' Recall\nFish dataset - GroupNet Scheme", wrap=True)
axs[2].set_xlabel("Discriminator's epoch")
axs[2].set_ylabel("Recall")
axs[2].legend(loc='lower right', bbox_to_anchor=(0.99, 0.0))
axs[2].grid(False)
plt.tight_layout()
plt.show()



fig, axs = plt.subplots(1, 1, figsize=(8, 5))

axs.plot(epochs, base_dus, label='Base-Rigid')
axs.fill_between(epochs, base_dus - base_dus_std, base_dus + base_dus_std, alpha=0.2)
axs.plot(epochs, closest_dus, label='Base-Smooth')
axs.fill_between(epochs, closest_dus - closest_dus_std, closest_dus + closest_dus_std, alpha=0.2)
axs.plot(epochs, gan10_dus, label='10GAN',color='green')
axs.fill_between(epochs, gan10_dus - gan10_dus_std, gan10_dus + gan10_dus_std, alpha=0.2,color='green')
axs.plot(epochs, gan100_dus, label='100GAN',color='red')
axs.fill_between(epochs, gan100_dus - gan100_dus_std, gan100_dus + gan100_dus_std, alpha=0.2,color='red')
axs.plot(epochs, gan250_dus, label='250GAN',color='purple')
axs.fill_between(epochs, gan250_dus - gan250_dus_std, gan250_dus + gan250_dus_std, alpha=0.2,color='purple')
axs.set_title("Discriminator's uncontrolled Agents' Scores\nFish dataset - GroupNet Scheme", wrap=True)
axs.set_xlabel("Discriminator's epoch")
axs.set_ylabel("Score")
axs.legend(loc='lower right', bbox_to_anchor=(0.99, 0.2))
axs.grid(False)

plt.tight_layout()
plt.show()



fig, axs = plt.subplots(1, 3, figsize=(20, 5))

axs[0].plot(epochs, base_acc, label='Base-Rigid')
axs[0].fill_between(epochs, base_acc - base_acc_std, base_acc + base_acc_std, alpha=0.2)
axs[0].plot(epochs, closest_acc, label='Base-Smooth')
axs[0].fill_between(epochs, closest_acc - closest_acc_std, closest_acc + closest_acc_std, alpha=0.2)
axs[0].plot(epochs, gan10_acc, label='10GAN' ,color='green')
axs[0].fill_between(epochs, gan10_acc - gan10_acc_std, gan10_acc + gan10_acc_std, alpha=0.2,color='green')
axs[0].plot(epochs, gan100_acc, label='100GAN',color='red')
axs[0].fill_between(epochs, gan100_acc - gan100_acc_std, gan100_acc + gan100_acc_std, alpha=0.2,color='red')
axs[0].plot(epochs, gan250_acc, label='250GAN',color='purple')
axs[0].fill_between(epochs, gan250_acc - gan250_acc_std, gan250_acc + gan250_acc_std, alpha=0.2,color='purple')
axs[0].set_title("Agents' Accuracy\nFish dataset - GroupNet Scheme", wrap=True)
axs[0].set_xlabel("Discriminator's epoch")
axs[0].set_ylabel("Accuracy")
axs[0].legend()
axs[0].grid(False)

axs[1].plot(epochs, base_per, label='Base-Rigid')
axs[1].fill_between(epochs, base_per - base_per_std, base_per + base_per_std, alpha=0.2)
axs[1].plot(epochs, closest_per, label='Base-Smooth')
axs[1].fill_between(epochs, closest_per - closest_per_std, closest_per + closest_per_std, alpha=0.2)
axs[1].plot(epochs, gan10_per, label='10GAN',color='green')
axs[1].fill_between(epochs, gan10_per - gan10_per_std, gan10_per + gan10_per_std, alpha=0.2,color='green')
axs[1].plot(epochs, gan100_per, label='100GAN',color='red')
axs[1].fill_between(epochs, gan100_per - gan100_per_std, gan100_per + gan100_per_std, alpha=0.2,color='red')
axs[1].plot(epochs, gan250_per, label='250GAN',color='purple')
axs[1].fill_between(epochs, gan250_per - gan250_per_std, gan250_per + gan250_per_std, alpha=0.2,color='purple')
axs[1].set_title("Agents' Precision\nFish dataset - GroupNet Scheme", wrap=True)
axs[1].set_xlabel("Discriminator's epoch")
axs[1].set_ylabel("Precision")
axs[1].legend()
axs[1].grid(False)

axs[2].plot(epochs, base_f1, label='Base-Rigid')
axs[2].fill_between(epochs, base_f1 - base_f1_std, base_f1 + base_f1_std, alpha=0.2)
axs[2].plot(epochs, closest_f1, label='Base-Smooth')
axs[2].fill_between(epochs, closest_f1 - closest_f1_std, closest_f1 +closest_f1_std, alpha=0.2)
axs[2].plot(epochs, gan10_f1, label='10GAN',color='green')
axs[2].fill_between(epochs, gan10_f1 - gan10_f1_std, gan10_f1 + gan10_f1_std, alpha=0.2,color='green')
axs[2].plot(epochs, gan100_f1, label='100GAN',color='red')
axs[2].fill_between(epochs, gan100_f1 - gan100_f1_std, gan100_f1 +gan100_f1_std, alpha=0.2,color='red')
axs[2].plot(epochs, gan250_f1, label='250GAN',color='purple')
axs[2].fill_between(epochs, gan250_f1 - gan250_f1_std, gan250_f1 + gan250_f1_std, alpha=0.2,color='purple')

axs[2].set_title("Agents' F1 Score\nFish dataset - GroupNet Scheme", wrap=True)
axs[2].set_xlabel("Discriminator's epoch")
axs[2].set_ylabel("F1")
axs[2].legend(loc='lower right', bbox_to_anchor=(0.99, 0.0))
axs[2].grid(False)
plt.tight_layout()
plt.show()






#####SAMPLER#################
#####SAMPLER#################
# Data for Baseline
base_dus = np.array([0.54, 0.60, 0.69, 0.77 , 0.81, 0.84])
base_dus_std = np.array([0.01, 0.06, 0.07, 0.06,0.04, 0.04])

base_dcs =np.array([0.49, 0.43, 0.35, 0.36, 0.36, 0.18])
base_dcs_std = np.array([0.02, 0.03, 0.06,0.05, 0.08, 0.06])

base_recall = np.array([0.73, 0.97, 0.99, 0.99, 0.97, 1])
base_recall_std = np.array([0.19,0.05, 0.01,0.03, 0.05,0, ])

base_acc = np.array([0.56, 0.68, 0.69, 0.69, 0.68, 0.7])
base_acc_std =  np.array([0.23, 0.16, 0.15, 0.15, 0.16, 0.15])

base_per =  np.array([0.56, 0.64, 0.65, 0.64, 0.64, 0.65])
base_per_std =  np.array([0.19, 0.14, 0.13, 0.13, 0.14, 0.13])

base_f1 = np.array([0.63, 0.76, 0.77, 0.77, 0.76, 0.78])
base_f1_std =  np.array([0.19, 0.1, 0.09, 0.09, 0.1, 0.09])




# Data for Closest
closest_dus =np.array([0.73, 0.72, 0.76, 0.88, 0.92, 0.97])
closest_dus_std = np.array([0.01, 0.03, 0.07, 0.05,0.02, 0.01])

closest_dcs = np.array([0.7, 0.65, 0.51, 0.42, 0.18, 0.04])
closest_dcs_std = np.array([0.03, 0.01, 0.08,0.06, 0.10, 0.07])

closest_recall = np.array([0.4, 0.4, 0.66, 0.87, 0.99, 0.99])
closest_recall_std = np.array([0.3,0.3, 0.22,0.12, 0.03,0.01, ])

closest_acc = np.array([0.4,0.4, 0.52, 0.63 , 0.69, 0.69])
closest_acc_std =  np.array([0.3, 0.3, 0.26, 0.19, 0.15, 0.15])

closest_per =  np.array([0.4, 0.4, 0.52, 0.61, 0.64, 0.65])
closest_per_std =  np.array([0.3, 0.3, 0.23, 0.15, 0.13, 0.13])

closest_f1 = np.array([0.4, 0.4, 0.57, 0.71, 0.77, 0.77])
closest_f1_std =  np.array([0.3, 0.3, 0.23, 0.14, 0.09, 0.09])



gan10_dcs_S = np.array([0.41, 0.34, 0.32, 0.28, 0.25, 0.26])
gan10_dcs_std_S = np.array([0.02, 0.04, 0.04, 0.04, 0.04, 0.04])

gan10_dus_S = np.array([0.51, 0.53, 0.51, 0.50, 0.48, 0.50])
gan10_dus_std_S = np.array([0.02, 0.03, 0.04, 0.04, 0.04, 0.01])

gan10_recall_S = np.array([0.99, 1, 0.99, 1, 1, 1])
gan10_recall_std_S = np.array([0.01, 0.0, 0.02, 0, 0, 0])

gan10_acc = np.array([0.63, 0.66, 0.63, 0.57, 0.58, 0.61])
gan10_acc_std =  np.array([0.13, 0.14, 0.12, 0.1, 0.11, 0.12])

gan10_per =  np.array([0.59, 0.61, 0.58, 0.54, 0.55, 0.58])
gan10_per_std =  np.array([0.1, 0.12, 0.1, 0.08, 0.08, 0.09])

gan10_f1 = np.array([0.73, 0.75, 0.73, 0.7, 0.71, 0.72])
gan10_f1_std =  np.array([0.07, 0.08, 0.07, 0.06, 0.06, 0.09])




gan100_dcs_S = np.array([0.48, 0.43, 0.47, 0.41, 0.4, 0.41])
gan100_dcs_std_S = np.array([0.01, 0.02, 0.03, 0.03, 0.04, 0.04])

gan100_dus_S =np.array([0.53, 0.48, 0.53, 0.49, 0.49, 0.54])
gan100_dus_std_S = np.array([0.02, 0.02, 0.03, 0.03, 0.04, 0.03])

gan100_recall_S = np.array([0.88, 0.99, 0.88, 1, 1, 0.99])
gan100_recall_std_S =np.array([0.14, 0.01, 0.14, 0.0, 0.07, 0.01])

gan100_acc = np.array([0.62, 0.55, 0.62, 0.57, 0.59, 0.66])
gan100_acc_std =  np.array([0.18, 0.07, 0.18, 0.1, 0.11, 0.13])

gan100_per =  np.array([0.6, 0.53, 0.6, 0.54, 0.56, 0.61])
gan100_per_std =  np.array([0.15, 0.05, 0.15, 0.08, 0.08, 0.11])

gan100_f1 = np.array([0.7, 0.69, 0.71, 0.7, 0.71, 0.75])
gan100_f1_std =  np.array([0.13, 0.04, 0.13, 0.06, 0.06, 0.08])


# Data for 10GM-no mission
gm10_nomission_S = np.array([0.41, 0.35, 0.32, 0.28, 0.26, 0.26])
gm10_nomission_std_S = np.array([0.02, 0.03, 0.04, 0.04, 0.04, 0.03])

# Data for 100GM-no mission
gm100_nomission_S = np.array([0.49, 0.43, 0.47, 0.42, 0.41, 0.42])
gm100_nomission_std_S = np.array([0.02, 0.02, 0.03, 0.02, 0.03, 0.03])


fig, axs = plt.subplots(1, 3, figsize=(20, 5))

axs[0].plot(epochs, base_dcs, label='Base-Rigid')
axs[0].fill_between(epochs, base_dcs - base_dcs_std, base_dcs + base_dcs_std, alpha=0.2)
axs[0].plot(epochs, closest_dcs, label='Base-Smooth')
axs[0].fill_between(epochs, closest_dcs - closest_dcs_std, closest_dcs + closest_dcs_std, alpha=0.2)
axs[0].plot(epochs, gan10_dcs_S, label='10GAN',color='green')
axs[0].fill_between(epochs, gan10_dcs_S - gan10_dcs_std_S, gan10_dcs_S + gan10_dcs_std_S, alpha=0.2,color='green')
axs[0].plot(epochs, gan100_dcs_S, label='100GAN',color='red')
axs[0].fill_between(epochs, gan100_dcs_S - gan100_dcs_std_S, gan100_dcs_S + gan100_dcs_std_S, alpha=0.2,color='red')
axs[0].set_title("(a) Discriminator's controlled Agents' Scores\nFish dataset - Sampler Scheme", wrap=True)
axs[0].set_xlabel("Discriminator's epoch")
axs[0].set_ylabel("Score")
axs[0].legend()
axs[0].grid(False)

# axs[1].plot(epochs, base_dus, label='Base-Rigid')
# axs[1].fill_between(epochs, base_dus - base_dus_std, base_dus + base_dus_std, alpha=0.2)
# axs[1].plot(epochs, closest_dus, label='Base-Smooth')
# axs[1].fill_between(epochs, closest_dus - closest_dus_std, closest_dus + closest_dus_std, alpha=0.2)
# axs[1].plot(epochs, gan10_dus_S, label='10GAN',color='green')
# axs[1].fill_between(epochs, gan10_dus_S - gan10_dus_std_S, gan10_dus_S + gan10_dus_std_S, alpha=0.2,color='green')
# axs[1].plot(epochs, gan100_dus_S, label='100GAN',color='red')
# axs[1].fill_between(epochs, gan100_dus_S - gan100_dus_std_S, gan100_dus_S + gan100_dus_std_S, alpha=0.2,color='red')
# axs[1].set_title("Discriminator's uncntrolled Agents' Scores (DUS)\nFish dataset - Sampler Scheme", wrap=True)
# axs[1].set_xlabel("Discriminator's epoch")
# axs[1].set_ylabel("Score")
# axs[1].legend()
# axs[1].grid(False)

axs[1].plot(epochs, gm10_nomission_S, label='10SM-no mission',  color='green')
axs[1].fill_between(epochs, gm10_nomission_S - gm10_nomission_std_S, gm10_nomission_S + gm10_nomission_std_S, alpha=0.2,  color='green')

axs[1].plot(epochs, gm100_nomission_S, label='100SM-no mission',  color='red')
axs[1].fill_between(epochs, gm100_nomission_S - gm100_nomission_std, gm100_nomission_S + gm100_nomission_std_S, alpha=0.2,  color='red')

axs[1].set_title("(b) Controlled (No Mission) Agents' Scores\nFish dataset - Sampler Scheme")
axs[1].set_xlabel("Discriminator's epoch")
axs[1].set_ylabel("Score")
axs[1].legend()
axs[1].grid(False)

axs[2].plot(epochs, base_recall, label='Base-Rigid')
axs[2].fill_between(epochs, base_recall - base_recall_std, base_recall + base_recall_std, alpha=0.2)
axs[2].plot(epochs, closest_recall, label='Base-Smooth')
axs[2].fill_between(epochs, closest_recall - closest_recall_std, closest_recall + closest_recall_std, alpha=0.2)
axs[2].plot(epochs, gan10_recall_S, label='10GAN',color='green')
axs[2].fill_between(epochs, gan10_recall_S - gan100_recall_std_S, gan10_recall_S + gan100_recall_std_S, alpha=0.2,color='green')
axs[2].plot(epochs, gan100_recall_S, label='100GAN',color='red')
axs[2].fill_between(epochs, gan100_recall_S - gan100_recall_std_S, gan100_recall_S + gan100_recall_std_S, alpha=0.2,color='red')
axs[2].set_title("(c) Agents' Recall\nFish dataset - Sampler Scheme", wrap=True)
axs[2].set_xlabel("Discriminator's epoch")
axs[2].set_ylabel("Recall")
axs[2].legend(loc='lower right', bbox_to_anchor=(0.99, 0.0))
axs[2].grid(False)
plt.tight_layout()
plt.show()



fig, axs = plt.subplots(1, 1, figsize=(8, 5))

axs.plot(epochs, base_dus, label='Base-Rigid')
axs.fill_between(epochs, base_dus - base_dus_std, base_dus + base_dus_std, alpha=0.2)
axs.plot(epochs, closest_dus, label='Base-Smooth')
axs.fill_between(epochs, closest_dus - closest_dus_std, closest_dus + closest_dus_std, alpha=0.2)
axs.plot(epochs, gan10_dus_S, label='10GAN',color='green')
axs.fill_between(epochs, gan10_dus_S - gan10_dus_std_S, gan10_dus_S + gan10_dus_std_S, alpha=0.2,color='green')
axs.plot(epochs, gan100_dus_S, label='100GAN',color='red')
axs.fill_between(epochs, gan100_dus_S - gan100_dus_std_S, gan100_dus_S + gan100_dus_std_S, alpha=0.2,color='red')
axs.set_title("Discriminator's uncontrolled Agents' Scores\nFish dataset - Sampler Scheme", wrap=True)
axs.set_xlabel("Discriminator's epoch")
axs.set_ylabel("Score")
axs.legend()
axs.grid(False)

plt.tight_layout()
plt.show()


fig, axs = plt.subplots(1, 3, figsize=(20, 5))

axs[0].plot(epochs, base_acc, label='Base-Rigid')
axs[0].fill_between(epochs, base_acc - base_acc_std, base_acc + base_acc_std, alpha=0.2)
axs[0].plot(epochs, closest_acc, label='Base-Smooth')
axs[0].fill_between(epochs, closest_acc - closest_acc_std, closest_acc + closest_acc_std, alpha=0.2)
axs[0].plot(epochs, gan10_acc, label='10GAN' ,color='green')
axs[0].fill_between(epochs, gan10_acc - gan10_acc_std, gan10_acc + gan10_acc_std, alpha=0.2,color='green')
axs[0].plot(epochs, gan100_acc, label='100GAN',color='red')
axs[0].fill_between(epochs, gan100_acc - gan100_acc_std, gan100_acc + gan100_acc_std, alpha=0.2,color='red')
axs[0].set_title("Agents' Accuracy\nFish dataset - Sampler Scheme", wrap=True)
axs[0].set_xlabel("Discriminator's epoch")
axs[0].set_ylabel("Accuracy")
axs[0].legend()
axs[0].grid(False)

axs[1].plot(epochs, base_per, label='Base-Rigid')
axs[1].fill_between(epochs, base_per - base_per_std, base_per + base_per_std, alpha=0.2)
axs[1].plot(epochs, closest_per, label='Base-Smooth')
axs[1].fill_between(epochs, closest_per - closest_per_std, closest_per + closest_per_std, alpha=0.2)
axs[1].plot(epochs, gan10_per, label='10GAN',color='green')
axs[1].fill_between(epochs, gan10_per - gan10_per_std, gan10_per + gan10_per_std, alpha=0.2,color='green')
axs[1].plot(epochs, gan100_per, label='100GAN',color='red')
axs[1].fill_between(epochs, gan100_per - gan100_per_std, gan100_per + gan100_per_std, alpha=0.2,color='red')
axs[1].set_title("Agents' Precision\nFish dataset - Sampler Scheme", wrap=True)
axs[1].set_xlabel("Discriminator's epoch")
axs[1].set_ylabel("Precision")
axs[1].legend()
axs[1].grid(False)

axs[2].plot(epochs, base_f1, label='Base-Rigid')
axs[2].fill_between(epochs, base_f1 - base_f1_std, base_f1 + base_f1_std, alpha=0.2)
axs[2].plot(epochs, closest_f1, label='Base-Smooth')
axs[2].fill_between(epochs, closest_f1 - closest_f1_std, closest_f1 +closest_f1_std, alpha=0.2)
axs[2].plot(epochs, gan10_f1, label='10GAN',color='green')
axs[2].fill_between(epochs, gan10_f1 - gan10_f1_std, gan10_f1 + gan10_f1_std, alpha=0.2,color='green')
axs[2].plot(epochs, gan100_f1, label='100GAN',color='red')
axs[2].fill_between(epochs, gan100_f1 - gan100_f1_std, gan100_f1 +gan100_f1_std, alpha=0.2,color='red')

axs[2].set_title("Agents' F1 Score\nFish dataset - Sampler Scheme", wrap=True)
axs[2].set_xlabel("Discriminator's epoch")
axs[2].set_ylabel("F1")
axs[2].legend(loc='lower right', bbox_to_anchor=(0.99, 0.0))
axs[2].grid(False)
plt.tight_layout()
plt.show()