import matplotlib.pyplot as plt

""" Plot the scaling plots for our abolition test. The graphs have compute (GPU hours) on the x-axis and validation loss on the y-axis."""

compute_for_small_model = [526 / 3600, 1258 / 3600, 2937 / 3600, 6031 / 3600]
compute_for_big_model = [632 / 3600, 1265 / 3600, 2423 / 3600, 5717 / 3600, 6553 / 3600]

val_loss_small_model = [5.78, 4.76, 3.98, 3.93]
val_loss_big_model = [4.22, 4.10, 4.02, 3.65, 3.48]

plt.cla(); plt.clf()
plt.plot(compute_for_small_model, val_loss_small_model, label='8M Params', color='blue')
plt.plot(compute_for_big_model, val_loss_big_model, label='20M Params', color='orange')
plt.xlabel('Compute (GPU Hours)')
plt.ylabel('Validation Loss')
plt.title(f"Scaling Plot for 8M Param LLaMA Model vs 20M Param LLaMA Model")
plt.legend()
plt.savefig(f'graphs/scaling_plot_8M_20M.png')
