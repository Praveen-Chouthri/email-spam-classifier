"""
Balancing metrics component for tracking class balancing effectiveness.

This module provides the BalancingMetrics component that tracks distribution
changes, calculates improvement metrics, and generates detailed reports on
class balancing effectiveness for spam detection.
"""
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class DistributionMetrics:
    """Metrics for class distribution analysis."""
    total_samples: int
    spam_count: int
    legitimate_count: int
    spam_ratio: float
    legitimate_ratio: float
    imbalance_ratio: float  # minority_count / majority_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'total_samples': self.total_samples,
            'spam_count': self.spam_count,
            'legitimate_count': self.legitimate_count,
            'spam_ratio': self.spam_ratio,
            'legitimate_ratio': self.legitimate_ratio,
            'imbalance_ratio': self.imbalance_ratio
        }


@dataclass
class PerformanceComparison:
    """Comparison of model performance before and after balancing."""
    original_accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    original_precision: float = 0.0
    balanced_precision: float = 0.0
    original_recall: float = 0.0
    balanced_recall: float = 0.0
    original_f1_score: float = 0.0
    balanced_f1_score: float = 0.0
    original_false_negative_rate: float = 0.0
    balanced_false_negative_rate: float = 0.0
    original_false_positive_rate: float = 0.0
    balanced_false_positive_rate: float = 0.0
    
    def calculate_improvements(self) -> Dict[str, float]:
        """Calculate improvement metrics."""
        return {
            'accuracy_improvement': self.balanced_accuracy - self.original_accuracy,
            'precision_improvement': self.balanced_precision - self.original_precision,
            'recall_improvement': self.balanced_recall - self.original_recall,
            'f1_improvement': self.balanced_f1_score - self.original_f1_score,
            'fnr_reduction': self.original_false_negative_rate - self.balanced_false_negative_rate,
            'fpr_change': self.balanced_false_positive_rate - self.original_false_positive_rate
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'original_metrics': {
                'accuracy': self.original_accuracy,
                'precision': self.original_precision,
                'recall': self.original_recall,
                'f1_score': self.original_f1_score,
                'false_negative_rate': self.original_false_negative_rate,
                'false_positive_rate': self.original_false_positive_rate
            },
            'balanced_metrics': {
                'accuracy': self.balanced_accuracy,
                'precision': self.balanced_precision,
                'recall': self.balanced_recall,
                'f1_score': self.balanced_f1_score,
                'false_negative_rate': self.balanced_false_negative_rate,
                'false_positive_rate': self.balanced_false_positive_rate
            },
            'improvements': self.calculate_improvements()
        }


@dataclass
class BalancingReport:
    """Comprehensive balancing effectiveness report."""
    timestamp: datetime
    balancing_method: str
    target_spam_ratio: float
    original_distribution: DistributionMetrics
    balanced_distribution: DistributionMetrics
    synthetic_samples_created: int
    processing_time: float
    performance_comparison: Optional[PerformanceComparison] = None
    recommendations: List[str] = field(default_factory=list)
    validation_passed: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'balancing_method': self.balancing_method,
            'target_spam_ratio': self.target_spam_ratio,
            'original_distribution': self.original_distribution.to_dict(),
            'balanced_distribution': self.balanced_distribution.to_dict(),
            'synthetic_samples_created': self.synthetic_samples_created,
            'processing_time': self.processing_time,
            'performance_comparison': self.performance_comparison.to_dict() if self.performance_comparison else None,
            'recommendations': self.recommendations,
            'validation_passed': self.validation_passed
        }


class BalancingMetrics:
    """
    Component for tracking and reporting class balancing effectiveness.
    
    This class provides comprehensive metrics tracking for class balancing operations,
    including distribution analysis, performance comparisons, and detailed reporting.
    """
    
    def __init__(self):
        """Initialize the BalancingMetrics component."""
        self.logger = logging.getLogger(__name__)
        
        # Distribution tracking
        self.original_distribution: Optional[DistributionMetrics] = None
        self.balanced_distribution: Optional[DistributionMetrics] = None
        
        # Performance tracking
        self.performance_comparison: Optional[PerformanceComparison] = None
        
        # Balancing metadata
        self.balancing_method: str = 'none'
        self.target_spam_ratio: float = 0.0
        self.synthetic_samples_created: int = 0
        self.processing_time: float = 0.0
        self.validation_passed: bool = True
        
        # Historical tracking
        self.balancing_history: List[BalancingReport] = []
        
        self.logger.info("BalancingMetrics component initialized")
    
    def record_original_distribution(self, y: np.ndarray) -> None:
        """
        Record the original class distribution before balancing.
        
        Args:
            y: Target labels (0 for legitimate, 1 for spam)
        """
        if len(y) == 0:
            self.logger.warning("Empty target array provided for distribution recording")
            return
        
        self.original_distribution = self._calculate_distribution_metrics(y)
        
        self.logger.info(f"Original distribution recorded: "
                        f"{self.original_distribution.spam_count} spam, "
                        f"{self.original_distribution.legitimate_count} legitimate "
                        f"(spam ratio: {self.original_distribution.spam_ratio:.3f})")
    
    def record_balanced_distribution(self, y: np.ndarray) -> None:
        """
        Record the class distribution after balancing.
        
        Args:
            y: Balanced target labels (0 for legitimate, 1 for spam)
        """
        if len(y) == 0:
            self.logger.warning("Empty target array provided for balanced distribution recording")
            return
        
        self.balanced_distribution = self._calculate_distribution_metrics(y)
        
        # Calculate synthetic samples created
        if self.original_distribution:
            self.synthetic_samples_created = (
                self.balanced_distribution.total_samples - 
                self.original_distribution.total_samples
            )
        
        self.logger.info(f"Balanced distribution recorded: "
                        f"{self.balanced_distribution.spam_count} spam, "
                        f"{self.balanced_distribution.legitimate_count} legitimate "
                        f"(spam ratio: {self.balanced_distribution.spam_ratio:.3f})")
        
        if self.synthetic_samples_created > 0:
            self.logger.info(f"Synthetic samples created: {self.synthetic_samples_created}")
    
    def record_balancing_metadata(self, 
                                method: str, 
                                target_ratio: float, 
                                processing_time: float,
                                validation_passed: bool = True) -> None:
        """
        Record balancing operation metadata.
        
        Args:
            method: Balancing method used ('smote', 'class_weights', 'both')
            target_ratio: Target spam ratio
            processing_time: Time taken for balancing operation
            validation_passed: Whether validation checks passed
        """
        self.balancing_method = method
        self.target_spam_ratio = target_ratio
        self.processing_time = processing_time
        self.validation_passed = validation_passed
        
        self.logger.info(f"Balancing metadata recorded: method={method}, "
                        f"target_ratio={target_ratio:.3f}, "
                        f"processing_time={processing_time:.2f}s")
    
    def record_performance_metrics(self, 
                                 original_results: Dict[str, float], 
                                 balanced_results: Dict[str, float]) -> None:
        """
        Record performance metrics for comparison.
        
        Args:
            original_results: Performance metrics from unbalanced model
            balanced_results: Performance metrics from balanced model
        """
        self.performance_comparison = PerformanceComparison(
            original_accuracy=original_results.get('accuracy', 0.0),
            balanced_accuracy=balanced_results.get('accuracy', 0.0),
            original_precision=original_results.get('precision', 0.0),
            balanced_precision=balanced_results.get('precision', 0.0),
            original_recall=original_results.get('recall', 0.0),
            balanced_recall=balanced_results.get('recall', 0.0),
            original_f1_score=original_results.get('f1_score', 0.0),
            balanced_f1_score=balanced_results.get('f1_score', 0.0),
            original_false_negative_rate=original_results.get('false_negative_rate', 0.0),
            balanced_false_negative_rate=balanced_results.get('false_negative_rate', 0.0),
            original_false_positive_rate=original_results.get('false_positive_rate', 0.0),
            balanced_false_positive_rate=balanced_results.get('false_positive_rate', 0.0)
        )
        
        improvements = self.performance_comparison.calculate_improvements()
        self.logger.info(f"Performance comparison recorded:")
        self.logger.info(f"  - Accuracy improvement: {improvements['accuracy_improvement']:.3f}")
        self.logger.info(f"  - Recall improvement: {improvements['recall_improvement']:.3f}")
        self.logger.info(f"  - FNR reduction: {improvements['fnr_reduction']:.3f}")
    
    def calculate_improvement_metrics(self, 
                                    original_results: Dict[str, float], 
                                    balanced_results: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate comprehensive improvement metrics comparing before/after balancing performance.
        
        Args:
            original_results: Performance metrics from unbalanced model
            balanced_results: Performance metrics from balanced model
            
        Returns:
            Dictionary with improvement metrics and analysis
        """
        # Record performance metrics
        self.record_performance_metrics(original_results, balanced_results)
        
        if not self.performance_comparison:
            self.logger.error("Performance comparison not available")
            return {}
        
        improvements = self.performance_comparison.calculate_improvements()
        
        # Calculate distribution improvements
        distribution_improvements = {}
        if self.original_distribution and self.balanced_distribution:
            distribution_improvements = {
                'spam_ratio_change': (
                    self.balanced_distribution.spam_ratio - 
                    self.original_distribution.spam_ratio
                ),
                'imbalance_ratio_improvement': (
                    self.balanced_distribution.imbalance_ratio - 
                    self.original_distribution.imbalance_ratio
                ),
                'samples_added': self.synthetic_samples_created,
                'samples_added_percentage': (
                    self.synthetic_samples_created / self.original_distribution.total_samples * 100
                    if self.original_distribution.total_samples > 0 else 0.0
                )
            }
        
        # Calculate effectiveness score
        effectiveness_score = self._calculate_effectiveness_score(improvements)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(improvements, distribution_improvements)
        
        return {
            'performance_improvements': improvements,
            'distribution_improvements': distribution_improvements,
            'effectiveness_score': effectiveness_score,
            'recommendations': recommendations,
            'balancing_successful': effectiveness_score > 0.5,
            'target_achieved': self._check_target_achievement()
        }
    
    def generate_balancing_report(self) -> str:
        """
        Generate a detailed balancing report with statistics and recommendations.
        
        Returns:
            Formatted report string
        """
        if not self.original_distribution:
            return "No balancing data available for report generation."
        
        # Create report object
        report = BalancingReport(
            timestamp=datetime.now(),
            balancing_method=self.balancing_method,
            target_spam_ratio=self.target_spam_ratio,
            original_distribution=self.original_distribution,
            balanced_distribution=self.balanced_distribution or self.original_distribution,
            synthetic_samples_created=self.synthetic_samples_created,
            processing_time=self.processing_time,
            performance_comparison=self.performance_comparison,
            validation_passed=self.validation_passed
        )
        
        # Generate recommendations
        if self.performance_comparison:
            improvements = self.performance_comparison.calculate_improvements()
            distribution_improvements = {}
            if self.balanced_distribution:
                distribution_improvements = {
                    'spam_ratio_change': (
                        self.balanced_distribution.spam_ratio - 
                        self.original_distribution.spam_ratio
                    )
                }
            report.recommendations = self._generate_recommendations(improvements, distribution_improvements)
        
        # Add to history
        self.balancing_history.append(report)
        
        # Generate formatted report
        return self._format_report(report)
    
    def get_balancing_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current balancing metrics.
        
        Returns:
            Dictionary with balancing summary
        """
        summary = {
            'balancing_applied': self.balanced_distribution is not None,
            'method_used': self.balancing_method,
            'target_spam_ratio': self.target_spam_ratio,
            'processing_time': self.processing_time,
            'validation_passed': self.validation_passed
        }
        
        if self.original_distribution:
            summary['original_distribution'] = self.original_distribution.to_dict()
        
        if self.balanced_distribution:
            summary['balanced_distribution'] = self.balanced_distribution.to_dict()
            summary['synthetic_samples_created'] = self.synthetic_samples_created
        
        if self.performance_comparison:
            summary['performance_comparison'] = self.performance_comparison.to_dict()
        
        return summary
    
    def export_metrics_to_json(self, file_path: str) -> None:
        """
        Export balancing metrics to JSON file.
        
        Args:
            file_path: Path to save the JSON file
        """
        try:
            summary = self.get_balancing_summary()
            summary['export_timestamp'] = datetime.now().isoformat()
            
            with open(file_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"Balancing metrics exported to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics to {file_path}: {str(e)}")
    
    def _calculate_distribution_metrics(self, y: np.ndarray) -> DistributionMetrics:
        """Calculate distribution metrics from target labels."""
        unique_labels, counts = np.unique(y, return_counts=True)
        
        total_samples = len(y)
        spam_count = 0
        legitimate_count = 0
        
        for label, count in zip(unique_labels, counts):
            if label == 1:  # Spam
                spam_count = int(count)
            else:  # Legitimate
                legitimate_count = int(count)
        
        spam_ratio = spam_count / total_samples if total_samples > 0 else 0.0
        legitimate_ratio = legitimate_count / total_samples if total_samples > 0 else 0.0
        
        # Calculate imbalance ratio (minority / majority)
        minority_count = min(spam_count, legitimate_count)
        majority_count = max(spam_count, legitimate_count)
        imbalance_ratio = minority_count / majority_count if majority_count > 0 else 0.0
        
        return DistributionMetrics(
            total_samples=total_samples,
            spam_count=spam_count,
            legitimate_count=legitimate_count,
            spam_ratio=spam_ratio,
            legitimate_ratio=legitimate_ratio,
            imbalance_ratio=imbalance_ratio
        )
    
    def _calculate_effectiveness_score(self, improvements: Dict[str, float]) -> float:
        """Calculate overall effectiveness score for balancing."""
        # Weight different metrics based on importance for spam detection
        weights = {
            'recall_improvement': 0.4,  # Most important for spam detection
            'fnr_reduction': 0.3,       # Directly related to missing spam
            'f1_improvement': 0.2,      # Overall performance
            'accuracy_improvement': 0.1  # General improvement
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in improvements:
                # Normalize improvements to 0-1 scale
                normalized_improvement = max(0, min(1, improvements[metric] + 0.5))
                score += normalized_improvement * weight
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _check_target_achievement(self) -> bool:
        """Check if balancing achieved the target spam ratio."""
        if not self.balanced_distribution or self.target_spam_ratio == 0.0:
            return False
        
        tolerance = 0.05  # 5% tolerance
        actual_ratio = self.balanced_distribution.spam_ratio
        
        return abs(actual_ratio - self.target_spam_ratio) <= tolerance
    
    def _generate_recommendations(self, 
                                improvements: Dict[str, float], 
                                distribution_improvements: Dict[str, float]) -> List[str]:
        """Generate recommendations based on balancing results."""
        recommendations = []
        
        # Check false negative rate improvement
        fnr_reduction = improvements.get('fnr_reduction', 0.0)
        if fnr_reduction > 0.02:  # 2% improvement
            recommendations.append("Excellent FNR reduction achieved. Consider using balanced model in production.")
        elif fnr_reduction > 0.01:  # 1% improvement
            recommendations.append("Good FNR reduction. Monitor performance on new data before deployment.")
        else:
            recommendations.append("Limited FNR improvement. Consider adjusting target ratio or trying different balancing method.")
        
        # Check overall performance
        accuracy_improvement = improvements.get('accuracy_improvement', 0.0)
        if accuracy_improvement < -0.02:  # 2% decrease
            recommendations.append("Significant accuracy decrease detected. Consider reverting to unbalanced model.")
        elif accuracy_improvement < 0:
            recommendations.append("Slight accuracy decrease. Evaluate if FNR improvement justifies the trade-off.")
        
        # Check false positive rate
        fpr_change = improvements.get('fpr_change', 0.0)
        if fpr_change > 0.05:  # 5% increase
            recommendations.append("High false positive rate increase. Consider adjusting class weights or target ratio.")
        
        # Check target achievement
        if self._check_target_achievement():
            recommendations.append("Target spam ratio successfully achieved.")
        else:
            recommendations.append("Target spam ratio not achieved. Consider adjusting SMOTE parameters.")
        
        # Check synthetic samples
        if self.synthetic_samples_created > 0:
            original_total = self.original_distribution.total_samples if self.original_distribution else 0
            if original_total > 0:
                synthetic_percentage = (self.synthetic_samples_created / original_total) * 100
                if synthetic_percentage > 100:
                    recommendations.append("Very high number of synthetic samples created. Validate sample quality.")
                elif synthetic_percentage > 50:
                    recommendations.append("Moderate number of synthetic samples created. Monitor for overfitting.")
        
        return recommendations
    
    def _format_report(self, report: BalancingReport) -> str:
        """Format the balancing report as a readable string."""
        lines = [
            "=" * 80,
            "CLASS BALANCING EFFECTIVENESS REPORT",
            "=" * 80,
            f"Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Method: {report.balancing_method.upper()}",
            f"Target Spam Ratio: {report.target_spam_ratio:.1%}",
            f"Processing Time: {report.processing_time:.2f} seconds",
            f"Validation: {'PASSED' if report.validation_passed else 'FAILED'}",
            "",
            "DISTRIBUTION ANALYSIS",
            "-" * 40,
            "Original Distribution:",
            f"  Total Samples: {report.original_distribution.total_samples:,}",
            f"  Spam: {report.original_distribution.spam_count:,} ({report.original_distribution.spam_ratio:.1%})",
            f"  Legitimate: {report.original_distribution.legitimate_count:,} ({report.original_distribution.legitimate_ratio:.1%})",
            f"  Imbalance Ratio: {report.original_distribution.imbalance_ratio:.3f}",
            "",
            "Balanced Distribution:",
            f"  Total Samples: {report.balanced_distribution.total_samples:,}",
            f"  Spam: {report.balanced_distribution.spam_count:,} ({report.balanced_distribution.spam_ratio:.1%})",
            f"  Legitimate: {report.balanced_distribution.legitimate_count:,} ({report.balanced_distribution.legitimate_ratio:.1%})",
            f"  Imbalance Ratio: {report.balanced_distribution.imbalance_ratio:.3f}",
            f"  Synthetic Samples Added: {report.synthetic_samples_created:,}",
            ""
        ]
        
        # Add performance comparison if available
        if report.performance_comparison:
            improvements = report.performance_comparison.calculate_improvements()
            lines.extend([
                "PERFORMANCE COMPARISON",
                "-" * 40,
                f"Accuracy: {report.performance_comparison.original_accuracy:.3f} → {report.performance_comparison.balanced_accuracy:.3f} ({improvements['accuracy_improvement']:+.3f})",
                f"Precision: {report.performance_comparison.original_precision:.3f} → {report.performance_comparison.balanced_precision:.3f} ({improvements['precision_improvement']:+.3f})",
                f"Recall: {report.performance_comparison.original_recall:.3f} → {report.performance_comparison.balanced_recall:.3f} ({improvements['recall_improvement']:+.3f})",
                f"F1-Score: {report.performance_comparison.original_f1_score:.3f} → {report.performance_comparison.balanced_f1_score:.3f} ({improvements['f1_improvement']:+.3f})",
                f"False Negative Rate: {report.performance_comparison.original_false_negative_rate:.3f} → {report.performance_comparison.balanced_false_negative_rate:.3f} ({-improvements['fnr_reduction']:+.3f})",
                f"False Positive Rate: {report.performance_comparison.original_false_positive_rate:.3f} → {report.performance_comparison.balanced_false_positive_rate:.3f} ({improvements['fpr_change']:+.3f})",
                ""
            ])
        
        # Add recommendations
        if report.recommendations:
            lines.extend([
                "RECOMMENDATIONS",
                "-" * 40
            ])
            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")
        
        lines.extend([
            "=" * 80,
            ""
        ])
        
        return "\n".join(lines)