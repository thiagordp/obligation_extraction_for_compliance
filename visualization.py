from src.visualization import ObligationFrequencyPlot, ObligationAnalysisVisualizer


task = "obligation_filtering"
dataset = "data/processed/obligations_analysis/GDPR/obligation_analysis_system/GDPR.json"

plotter = ObligationFrequencyPlot(dataset, confidence=0.95)
plotter.run(save_path="data/analysis/obligation_filtering.png")

visualizer = ObligationAnalysisVisualizer(dataset)
visualizer.plot_bar_chart()

#
# visualizer = ObligationAnalysisVisualizer(dataset)
# visualizer.plot_bar_chart()