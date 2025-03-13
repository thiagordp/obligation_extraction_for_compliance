import json
import logging
import re

import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

from nltk import PorterStemmer, word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from seaborn.algorithms import bootstrap

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class ObligationFrequencyPlot:
    """Process and visualize the frequency of obligation filtering classifications with confidence intervals."""

    def __init__(self, filename: str, confidence: float = 0.95):
        """
        Initialize the class with the given JSON file.

        :param filename: Path to the JSON file.
        :param confidence: Confidence level for intervals (default is 95%).
        """
        self.filename = filename
        self.confidence = confidence
        self.data = None
        self.stats = {}

    def load_data(self) -> None:
        """Load obligation classifications from the JSON file into a Pandas DataFrame."""
        try:
            with open(self.filename, "r", encoding="utf-8") as file:
                raw_data = json.load(file)

            classifications = []
            for entry in raw_data:
                if "potential_deontic" in entry:
                    for deontic in entry["potential_deontic"]:
                        if "filtering" in deontic and "output" in deontic["filtering"]:
                            classifications.append(deontic["filtering"]["output"]["classification"])

            if not classifications:
                raise ValueError("No classifications found in the JSON file.")

            # Store as Pandas DataFrame
            self.data = pd.DataFrame(classifications, columns=["classification"])

        except (json.JSONDecodeError, FileNotFoundError) as e:
            logging.error(f"Error loading JSON file: {e}")
            raise

    def compute_frequencies(self) -> None:
        """Compute frequency of each classification and calculate confidence intervals."""
        if self.data is None:
            raise ValueError("No data loaded. Run load_data() first.")

        freq_table = self.data["classification"].value_counts(normalize=True)
        total = len(self.data)

        # Compute confidence intervals using normal approximation
        ci_width = (1.96 if self.confidence == 0.95 else 2.58) * np.sqrt((freq_table * (1 - freq_table)) / total)

        self.stats = pd.DataFrame({
            "Frequency": freq_table,
            "Lower_CI": (freq_table - ci_width).clip(lower=0),  # No negative values
            "Upper_CI": (freq_table + ci_width).clip(upper=1)  # No values above 1
        }).reset_index().rename(columns={"index": "Classification"})

    def get_statistics(self) -> pd.DataFrame:
        """Return computed statistics as a DataFrame."""
        if len(self.stats.keys()) > 0:
            raise ValueError("Data not processed. Run compute_frequencies() first.")
        return self.stats

    def plot(self, save_path: Optional[str] = None) -> None:
        """
        Generate a bar chart of classification frequencies with confidence intervals.

        :param save_path: (Optional) Path to save the plot as an image.
        """
        if self.stats is None or self.stats.empty:
            raise ValueError("Data not processed. Run compute_frequencies() first.")

        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))

        # Plot bars with confidence intervals
        print(self.stats)
        plt.bar(self.stats["classification"], self.stats["Frequency"],
                yerr=[self.stats["Frequency"] - self.stats["Lower_CI"],
                      self.stats["Upper_CI"] - self.stats["Frequency"]],
                capsize=5, color="skyblue", alpha=0.8)

        plt.xlabel("Obligation Filtering Classification")
        plt.ylabel("Frequency")
        plt.title(f"Frequency of Obligation Filtering Classifications ({int(self.confidence * 100)}% CI)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            logging.info(f"Plot saved to {save_path}")

        plt.show()

    def run(self, save_path: Optional[str] = None) -> None:
        """
        Execute the full workflow: Load data, compute statistics, and plot.

        :param save_path: (Optional) Path to save the plot.
        """
        self.load_data()
        self.compute_frequencies()
        self.plot(save_path)


# Download required NLTK data if not available
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download('punkt_tab')


class ObligationAnalysisVisualizer:
    def __init__(self, file_path, top_k=10):
        """
        Initialize the visualizer.

        :param file_path: Path to the JSON file containing obligation analysis data.
        :param top_k: Number of top elements to display in the bar chart.
        """
        self.file_path = file_path
        self.top_k = top_k
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.data = self._load_data()
        self.df = self._process_data()

    def _load_data(self):
        """Load the JSON file."""
        with open(self.file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _clean_text(self, text):
        """
        Apply preprocessing to text:
        - Convert to lowercase
        - Remove leading/trailing stopwords
        - Lemmatize words (convert to base form)
        """
        if not text:
            return None

        text = text.lower()  # Convert to lowercase
        words = word_tokenize(text)  # Tokenize words

        # Remove stopwords at the start and end of the phrase
        while words and words[0] in self.stop_words:
            words.pop(0)
        while words and words[-1] in self.stop_words:
            words.pop()

        # Apply lemmatization to each word
        words = [self.lemmatizer.lemmatize(word) for word in words]

        return " ".join(words) if words else None

    def _process_data(self):
        """Extract and preprocess obligation classification data into a structured format."""
        records = []
        for entry in self.data:
            for analysis in entry.get("potential_deontic", []):
                for output in analysis.get("analysis", {}).get("output", []):
                    records.append({
                        "ObligationTypeClassification": self._clean_text(output.get("ObligationTypeClassification")),
                        "Addressee": self._clean_text(output.get("Addressees", [{}])[0].get("value")),
                        "Predicate": self._clean_text(output.get("Predicate", {}).get("value")),
                        "Target": self._clean_text(output.get("Targets", [{}])[0].get("value")),
                        "Specification": self._clean_text(output.get("Specifications", [{}])[0].get("value")),
                        "Pre-Condition": self._clean_text(output.get("Pre-Conditions", [{}])[0].get("value")),
                        "Beneficiaries": self._clean_text(output.get("Beneficiaries", [{}])[0].get("value")),
                    })
        return pd.DataFrame(records)

    def plot_frequency(self, column):
        """
        Plot the top K most frequent values in a given column with confidence intervals.

        :param column: Column name to analyze.
        """
        if column not in self.df.columns:
            print(f"Column '{column}' not found in dataset.")
            return

        counts = self.df[column].dropna().value_counts().nlargest(self.top_k)
        labels = counts.index
        values = counts.values

        # Confidence intervals (95% normal approximation)
        conf_int = 1.96 * np.sqrt(values)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=values, y=labels, orient="h", ci=None, color="skyblue")
        plt.errorbar(values, range(len(labels)), xerr=conf_int, fmt="none", color="black", capsize=5)

        plt.xlabel("Frequency")
        plt.ylabel(column)
        plt.title(f"Top {self.top_k} Most Frequent {column} Values with Confidence Intervals")
        plt.show()

    def generate_all_charts(self):
        """Generate bar charts for all key elements."""
        for column in self.df.columns:
            self.plot_frequency(column)


import json
from scipy.stats import norm


class ObligationAnalysisVisualizer2:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_data()
        self.df = self.process_data()

    def load_data(self):
        """Loads the JSON data from the given file path."""
        with open(self.file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def process_data(self):
        """Extracts relevant obligation analysis data and structures it into a DataFrame."""
        records = []
        for entry in self.data:
            for potential in entry.get("potential_deontic", []):
                for obligation in potential.get("analysis", {}).get("output", []):
                    classification = obligation.get("ObligationTypeClassification", "Unknown")
                    if classification == "Obligation of Action":
                        continue
                    for key in ["Addressees", "Beneficiaries", "Predicate", "Targets", "Specifications",
                                "Pre-Conditions"]:
                        elements = obligation.get(key, [])

                        for element in elements:
                            if type(element) is not dict:
                                print("not dict", element)
                                continue
                            records.append(
                                {
                                    "Classification": classification,
                                    "Element": key,
                                    "Extraction Method": element.get("extraction_method", "Unknown"),
                                }
                            )
        return pd.DataFrame(records)

    def compute_confidence_intervals(self, df_grouped):
        """Computes the confidence interval for each group."""
        df_grouped["CI_Lower"] = df_grouped["Count"] - norm.ppf(0.975) * np.sqrt(df_grouped["Count"])
        df_grouped["CI_Upper"] = df_grouped["Count"] + norm.ppf(0.975) * np.sqrt(df_grouped["Count"])
        return df_grouped

    def plot_bar_chart(self):
        """Generates a bar chart comparing extraction method frequencies by obligation element type."""
        df_grouped = self.df.groupby(["Element", "Extraction Method"]).size().reset_index(name="Count")
        df_grouped = self.compute_confidence_intervals(df_grouped)

        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=df_grouped,
            x="Element",
            y="Count",
            hue="Extraction Method",
            ci=None,
        )

        # Add error bars for confidence intervals
        for index, row in df_grouped.iterrows():
            plt.errorbar(
                x=row["Element"],
                y=row["Count"],
                fmt="none",
                capsize=5,
                color="black",
            )

        plt.title("Comparison of Extraction Methods by Obligation Element")
        plt.xlabel("Obligation Element Type")
        plt.ylabel("Frequency")
        plt.legend(title="Extraction Method")
        plt.xticks(rotation=45)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()
