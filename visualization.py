from src.visualization import ObligationFrequencyPlot, ObligationAnalysisVisualizer


task = "obligation_filtering"
dataset_name="AI_Act"

dataset = f"data/processed/obligations_analysis/{dataset_name}/obligation_analysis_system/{dataset_name}.json"

plotter = ObligationFrequencyPlot(dataset, confidence=0.95)
plotter.run(save_path=f"data/analysis/{task}_{dataset_name}.pdf")

#visualizer = ObligationAnalysisVisualizer(dataset)
#visualizer.plot_bar_chart()

#
# visualizer = ObligationAnalysisVisualizer(dataset)
# visualizer.plot_bar_chart()