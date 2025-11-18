"""Fact Checking Agent: Validates and cleans facts from query outputs"""
import re
from typing import List, Dict

class FactCheckingAgent:
    def __init__(self, config):
        self.config = config

    def validate_facts(self, answers: List[str], metadata: List[Dict]) -> str:
        """Takes list of answer texts and metadata about sources. Cross-checks and cleans facts."""
        consolidated = {}
        issues = []

        # Extract numeric facts from each answer
        for answer, meta in zip(answers, metadata):
            numbers = re.findall(r"(\d+\.?\d*)\s*(m|meters|°C|bar|kg/m³|tons|ft|in)?", answer, re.IGNORECASE)
            for num, unit in numbers:
                key = unit.lower() if unit else "number"
                fvalue = float(num)
                if key not in consolidated:
                    consolidated[key] = [fvalue]
                else:
                    consolidated[key].append(fvalue)

        # Simple rule to detect anomalies: get mean and report values far from mean
        final_results = []
        for key, vals in consolidated.items():
            import statistics
            mean_val = statistics.mean(vals)
            for v in vals:
                if abs(v - mean_val) / mean_val > 0.2:  # >20% deviation flag
                    issues.append(f"Fact anomaly: {v} {key} deviates from mean {mean_val:.2f} {key}")

            # Use median as final fact value
            median_val = statistics.median(vals)
            final_results.append(f"Verified median {key}: {median_val}")

        # Format issues and facts into final report
        report = "Fact Checking Report:\n"
        if issues:
            report += "Issues detected:\n" + "\n".join(issues) + "\n"
        else:
            report += "No fact anomalies detected.\n"
        report += "Final verified facts:\n" + "\n".join(final_results)
        return report
