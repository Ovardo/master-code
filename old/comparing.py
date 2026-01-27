import gtsam
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from gtsam.symbol_shorthand import X, L
import gtsam.utils.plot as gtsam_plot
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

# TODO: currently not used

class VariableType(Enum):
    """Enum for variable types in SLAM"""
    POSE = "pose"
    LANDMARK = "landmark"


@dataclass
class CovarianceComparisonResult:
    """Results from comparing two covariance matrices"""
    variable_key: int
    variable_name: str
    cov_method1: np.ndarray
    cov_method2: np.ndarray
    max_diff: float
    relative_error: float
    is_equal: bool
    atol: float = 1e-10
    
    def __str__(self) -> str:
        return (f"{self.variable_name}:\n"
                f"  Max difference: {self.max_diff:.2e}\n"
                f"  Relative error: {self.relative_error:.2%}\n"
                f"  Equal (atol={self.atol})? {self.is_equal}")


class CovarianceAnalyzer:
    """
    Analyzes and compares covariances from factor graphs.
    Useful for validating covariance estimates for JCBB data association.
    """
    
    def __init__(self, factor_graph: gtsam.GaussianFactorGraph, 
                 solution: gtsam.VectorValues,
                 atol: float = 1e-10):
        """
        Initialize the analyzer.
        
        Args:
            factor_graph: The factor graph to analyze
            solution: The optimized solution
            atol: Absolute tolerance for comparison
        """
        self.gfg = factor_graph
        self.solution = solution
        self.atol = atol
        self.marginals = gtsam.Marginals(factor_graph, solution)
        
        # Compute information matrix and its inverse
        self.information_matrix, _ = factor_graph.hessian()
        self.full_covariance = np.linalg.inv(self.information_matrix)
        
        # Cache for computed covariances
        self._marginal_cache = {}
        self._joint_cache = {}
    
    def get_variable_type(self, key: int) -> VariableType:
        """Determine if a variable is a pose or landmark"""
        symbol = gtsam.Symbol(key)
        if symbol.chr() == ord('x'):
            return VariableType.POSE
        elif symbol.chr() == ord('l'):
            return VariableType.LANDMARK
        else:
            raise ValueError(f"Unknown variable type for key {key}")
    
    def get_variable_name(self, key: int) -> str:
        """Get human-readable name for a variable"""
        symbol = gtsam.Symbol(key)
        return f"{chr(symbol.chr()).upper()}{symbol.index()}"
    
    def get_marginal_covariance(self, key: int) -> np.ndarray:
        """Get marginal covariance for a single variable (cached)"""
        if key not in self._marginal_cache:
            self._marginal_cache[key] = self.marginals.marginalCovariance(key)
        return self._marginal_cache[key]
    
    def get_joint_marginal_covariance(self, keys: List[int]) -> np.ndarray:
        """Get joint marginal covariance for multiple variables (cached)"""
        keys_tuple = tuple(sorted(keys))
        if keys_tuple not in self._joint_cache:
            joint = self.marginals.jointMarginalCovariance(keys)
            self._joint_cache[keys_tuple] = joint.fullMatrix()
        return self._joint_cache[keys_tuple]
    
    def extract_from_full_covariance(self, keys: List[int]) -> np.ndarray:
        """Extract submatrix from full covariance matrix for given variables"""
        indices = []
        for key in keys:
            # Find the indices for this variable in the full matrix
            # This assumes variables are ordered in the matrix
            var_idx = self._get_variable_indices(key)
            indices.extend(var_idx)
        
        # Extract submatrix
        sub_cov = self.full_covariance[np.ix_(indices, indices)]
        return sub_cov
    
    def _get_variable_indices(self, key: int) -> List[int]:
        """Get the indices in the full matrix for a given variable"""
        # For Point2 variables, each has 2 dimensions
        all_keys = self.gfg.keyVector()
        key_index = all_keys.index(key)
        start_idx = key_index * 2
        return [start_idx, start_idx + 1]
    
    def compare_individual_covariances(self, keys: List[int]) -> List[CovarianceComparisonResult]:
        """
        Compare individual marginal covariances from different methods.
        
        Args:
            keys: List of variable keys to compare
            
        Returns:
            List of comparison results
        """
        results = []
        joint_cov = self.get_joint_marginal_covariance(keys)
        
        for i, key in enumerate(keys):
            # Method 1: From joint marginal
            
            cov_from_joint = joint_cov[i*2:(i+1)*2, i*2:(i+1)*2]
            
            # Method 2: From individual marginal
            cov_from_marginal = self.get_marginal_covariance(key)
            
            # Compare
            max_diff = np.max(np.abs(cov_from_joint - cov_from_marginal))
            rel_error = max_diff / (np.max(np.abs(cov_from_marginal)) + 1e-10)
            is_equal = np.allclose(cov_from_joint, cov_from_marginal, atol=self.atol)
            
            result = CovarianceComparisonResult(
                variable_key=key,
                variable_name=self.get_variable_name(key),
                cov_method1=cov_from_joint,
                cov_method2=cov_from_marginal,
                max_diff=max_diff,
                relative_error=rel_error,
                is_equal=is_equal,
                atol=self.atol
            )
            results.append(result)
        
        return results
    
    def compare_joint_vs_full_inverse(self, keys: List[int]) -> Dict[str, Any]:
        """
        Compare joint marginal covariance against full inverse of information matrix.
        
        Args:
            keys: Variables to include in comparison
            
        Returns:
            Dictionary with comparison metrics
        """
        joint_marginal = self.get_joint_marginal_covariance(keys)
        
        # Extract corresponding block from full covariance
        full_inverse_block = self.extract_from_full_covariance(keys)
        
        max_diff = np.max(np.abs(joint_marginal - full_inverse_block))
        frobenius_norm_diff = np.linalg.norm(joint_marginal - full_inverse_block, 'fro')
        is_equal = np.allclose(joint_marginal, full_inverse_block, atol=self.atol)
        
        return {
            'max_difference': max_diff,
            'frobenius_norm_diff': frobenius_norm_diff,
            'is_equal': is_equal,
            'joint_marginal_shape': joint_marginal.shape,
            'condition_number_joint': np.linalg.cond(joint_marginal),
            'condition_number_full': np.linalg.cond(full_inverse_block)
        }
    
    def compute_correlation_matrix(self, keys: List[int]) -> np.ndarray:
        """
        Compute correlation matrix from covariance matrix.
        Useful for understanding variable dependencies in JCBB.
        """
        cov = self.get_joint_marginal_covariance(keys)
        std_dev = np.sqrt(np.diag(cov))
        correlation = cov / np.outer(std_dev, std_dev)
        return correlation
    
    def analyze_information_gain(self, keys: List[int]) -> Dict[str, float]:
        """
        Analyze information gain (reduction in uncertainty) for variables.
        Useful for active SLAM and sensor planning.
        """
        results = {}
        
        for key in keys:
            cov = self.get_marginal_covariance(key)
            # Determinant represents volume of uncertainty ellipse
            det = np.linalg.det(cov)
            # Trace represents total variance
            trace = np.trace(cov)
            # Maximum eigenvalue represents worst-case uncertainty
            eigenvalues = np.linalg.eigvals(cov)
            max_eigenvalue = np.max(eigenvalues)
            
            var_name = self.get_variable_name(key)
            results[var_name] = {
                'determinant': det,
                'trace': trace,
                'max_eigenvalue': max_eigenvalue,
                'entropy': 0.5 * np.log((2 * np.pi * np.e) ** 2 * det)  # Differential entropy
            }
        
        return results


class SLAMVisualizer:
    """
    Visualizer for SLAM results with covariance ellipses.
    Supports both pose and landmark visualization.
    """
    
    def __init__(self, solution: gtsam.VectorValues, 
                 ground_truth: Optional[Dict] = None):
        """
        Initialize visualizer.
        
        Args:
            solution: Optimized solution from factor graph
            ground_truth: Optional ground truth for comparison
        """
        self.solution = solution
        self.ground_truth = ground_truth
        
        # Visual style configuration
        self.pose_color = 'red'
        self.landmark_color = 'blue'
        self.ground_truth_color = 'green'
        self.covariance_alpha = 0.3
        self.n_std = 2  # Number of standard deviations for ellipse
    
    def plot_2d_slam(self, analyzer: CovarianceAnalyzer, 
                      pose_keys: List[int], 
                      landmark_keys: List[int],
                      fig_size: Tuple[int, int] = (15, 10),
                      show_correlation: bool = False) -> plt.Figure:
        """
        Create comprehensive SLAM visualization.
        
        Args:
            analyzer: Covariance analyzer instance
            pose_keys: List of pose variable keys
            landmark_keys: List of landmark variable keys
            fig_size: Figure size
            show_correlation: Whether to show correlation matrix
            
        Returns:
            Matplotlib figure
        """
        n_subplots = 3 if show_correlation else 2
        fig, axes = plt.subplots(1, n_subplots, figsize=fig_size)
        
        if not show_correlation:
            axes = [axes[0], axes[1], None]
        
        # Plot 1: Solution with covariances
        self._plot_solution_with_covariances(
            axes[0], analyzer, pose_keys, landmark_keys,
            title="SLAM Solution with Uncertainty"
        )
        
        # Plot 2: Comparison if ground truth available
        if self.ground_truth:
            self._plot_error_analysis(
                axes[1], pose_keys, landmark_keys,
                title="Error Analysis"
            )
        else:
            # Just plot solution without ground truth
            self._plot_solution_with_covariances(
                axes[1], analyzer, pose_keys, landmark_keys,
                title="Solution (No Ground Truth)"
            )
        
        # Plot 3: Correlation matrix
        if show_correlation:
            all_keys = pose_keys + landmark_keys
            self._plot_correlation_matrix(
                axes[2], analyzer, all_keys,
                title="Correlation Matrix"
            )
        
        plt.tight_layout()
        return fig
    
    def _plot_solution_with_covariances(self, ax: plt.Axes,
                                       analyzer: CovarianceAnalyzer,
                                       pose_keys: List[int],
                                       landmark_keys: List[int],
                                       title: str = ""):
        """Plot solution with covariance ellipses"""
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)
        
        # Plot poses
        pose_positions = []
        for key in pose_keys:
            mean = self.solution.at(key)
            pose_positions.append(mean)
            cov = analyzer.get_marginal_covariance(key)
            
            # Plot pose
            ax.plot(mean[0], mean[1], 'o', color=self.pose_color, 
                   markersize=8, label='Pose' if key == pose_keys[0] else '')
            
            # Add covariance ellipse
            self._add_covariance_ellipse(ax, mean, cov, self.pose_color)
            
            # Add label
            ax.text(mean[0], mean[1] + 0.15, 
                   analyzer.get_variable_name(key),
                   ha='center', fontsize=9)
        
        # Plot trajectory
        if len(pose_positions) > 1:
            trajectory = np.array(pose_positions)
            ax.plot(trajectory[:, 0], trajectory[:, 1], 
                   'r-', alpha=0.5, linewidth=1)
        
        # Plot landmarks
        for key in landmark_keys:
            mean = self.solution.at(key)
            cov = analyzer.get_marginal_covariance(key)
            
            # Plot landmark
            ax.plot(mean[0], mean[1], '^', color=self.landmark_color,
                   markersize=10, label='Landmark' if key == landmark_keys[0] else '')
            
            # Add covariance ellipse
            self._add_covariance_ellipse(ax, mean, cov, self.landmark_color)
            
            # Add label
            ax.text(mean[0], mean[1] + 0.15,
                   analyzer.get_variable_name(key),
                   ha='center', fontsize=9)
        
        # Plot ground truth if available
        if self.ground_truth:
            # Plot ground truth poses
            gt_poses = np.array(self.ground_truth['poses'])
            ax.plot(gt_poses[:, 0], gt_poses[:, 1], 
                   'o--', color=self.ground_truth_color, 
                   alpha=0.5, markersize=6, 
                   label='Ground Truth', linewidth=1)
            
            # Plot ground truth landmarks
            gt_landmarks = np.array(self.ground_truth['landmarks'])
            ax.plot(gt_landmarks[:, 0], gt_landmarks[:, 1],
                   '^', color=self.ground_truth_color,
                   alpha=0.5, markersize=8)
        
        ax.legend(loc='best')
    
    def _plot_error_analysis(self, ax: plt.Axes,
                            pose_keys: List[int],
                            landmark_keys: List[int],
                            title: str = ""):
        """Plot error vectors and statistics"""
        ax.set_title(title)
        ax.set_xlabel('Variable')
        ax.set_ylabel('Error (m)')
        ax.grid(True, alpha=0.3)
        
        if not self.ground_truth:
            ax.text(0.5, 0.5, 'No ground truth available',
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        errors = []
        labels = []
        colors = []
        
        # Compute pose errors
        for i, key in enumerate(pose_keys):
            est_pose = self.solution.at(key)
            true_pose = self.ground_truth['poses'][i]
            error = np.linalg.norm(est_pose - true_pose)
            errors.append(error)
            labels.append(f'X{i+1}')
            colors.append(self.pose_color)
        
        # Compute landmark errors
        for i, key in enumerate(landmark_keys):
            est_landmark = self.solution.at(key)
            true_landmark = self.ground_truth['landmarks'][i]
            error = np.linalg.norm(est_landmark - true_landmark)
            errors.append(error)
            labels.append(f'L{i+1}')
            colors.append(self.landmark_color)
        
        # Create bar plot
        x_pos = np.arange(len(labels))
        bars = ax.bar(x_pos, errors, color=colors, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        
        # Add value labels on bars
        for bar, error in zip(bars, errors):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{error:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Add RMSE lines
        pose_rmse = np.sqrt(np.mean(np.square(errors[:len(pose_keys)])))
        landmark_rmse = np.sqrt(np.mean(np.square(errors[len(pose_keys):])))
        
        ax.axhline(y=pose_rmse, color=self.pose_color, linestyle='--',
                  alpha=0.5, label=f'Pose RMSE: {pose_rmse:.3f}')
        ax.axhline(y=landmark_rmse, color=self.landmark_color, linestyle='--',
                  alpha=0.5, label=f'Landmark RMSE: {landmark_rmse:.3f}')
        
        ax.legend(loc='best')
    
    def _plot_correlation_matrix(self, ax: plt.Axes,
                                analyzer: CovarianceAnalyzer,
                                keys: List[int],
                                title: str = ""):
        """Plot correlation matrix as heatmap"""
        correlation = analyzer.compute_correlation_matrix(keys)
        
        # Create labels
        labels = []
        for key in keys:
            var_name = analyzer.get_variable_name(key)
            labels.extend([f'{var_name}_x', f'{var_name}_y'])
        
        # Plot heatmap
        im = ax.imshow(correlation, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        
        # Set ticks and labels
        tick_positions = np.arange(len(labels))
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        
        ax.set_title(title)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Add grid
        ax.set_xticks(np.arange(len(labels) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(labels) + 1) - 0.5, minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    def _add_covariance_ellipse(self, ax: plt.Axes, 
                               mean: np.ndarray,
                               cov: np.ndarray,
                               color: str):
        """Add covariance ellipse to plot"""
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Get angle of major axis
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        
        # Width and height are 2*sqrt(eigenvalue) for n_std standard deviations
        width = 2 * self.n_std * np.sqrt(eigenvalues[0])
        height = 2 * self.n_std * np.sqrt(eigenvalues[1])
        
        # Create ellipse
        ellipse = Ellipse(mean, width, height, angle=angle,
                         facecolor=color, alpha=self.covariance_alpha,
                         edgecolor=color, linewidth=1)
        ax.add_patch(ellipse)


def create_analysis_report(analyzer: CovarianceAnalyzer,
                          pose_keys: List[int],
                          landmark_keys: List[int]) -> str:
    """
    Create a comprehensive text report of covariance analysis.
    
    Args:
        analyzer: Covariance analyzer instance
        pose_keys: List of pose variable keys
        landmark_keys: List of landmark variable keys
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 60)
    report.append("COVARIANCE ANALYSIS REPORT")
    report.append("=" * 60)
    
    all_keys = pose_keys + landmark_keys
    
    # Individual covariance comparison
    report.append("\n1. INDIVIDUAL COVARIANCE COMPARISON")
    report.append("-" * 40)
    
    comparisons = analyzer.compare_individual_covariances(all_keys)
    for comp in comparisons:
        report.append(str(comp))
        report.append("")
    
    # Joint vs Full inverse comparison
    report.append("\n2. JOINT MARGINAL vs FULL INVERSE COMPARISON")
    report.append("-" * 40)
    
    joint_comparison = analyzer.compare_joint_vs_full_inverse(all_keys)
    report.append(f"Max difference: {joint_comparison['max_difference']:.2e}")
    report.append(f"Frobenius norm difference: {joint_comparison['frobenius_norm_diff']:.2e}")
    report.append(f"Matrices equal? {joint_comparison['is_equal']}")
    report.append(f"Condition number (joint): {joint_comparison['condition_number_joint']:.2f}")
    report.append(f"Condition number (full): {joint_comparison['condition_number_full']:.2f}")
    
    # Information gain analysis
    report.append("\n3. INFORMATION GAIN ANALYSIS")
    report.append("-" * 40)
    
    info_gain = analyzer.analyze_information_gain(all_keys)
    for var_name, metrics in info_gain.items():
        report.append(f"\n{var_name}:")
        report.append(f"  Determinant: {metrics['determinant']:.2e}")
        report.append(f"  Trace: {metrics['trace']:.4f}")
        report.append(f"  Max eigenvalue: {metrics['max_eigenvalue']:.4f}")
        report.append(f"  Differential entropy: {metrics['entropy']:.4f}")
    
    report.append("\n" + "=" * 60)
    
    return "\n".join(report)


# Example usage function
def run_covariance_analysis_example():
    """Example of how to use the covariance analysis framework"""
    from data.data_generator_linear import SimulationConfig, RobotSimulatorR2, build_linear_factor_graph
    
    # Create simulation
    config = SimulationConfig(
        poses=[
            np.array([0.0, 0.0]),
            np.array([2.0, 0.0]),
            np.array([4.0, 0.0])
        ],
        landmarks=[
            np.array([2.0, 2.0]),
            np.array([4.0, 2.0])
        ],
        observations={
            0: [0],
            1: [0],
            2: [1]
        },
        prior_noise_sim=np.array([0, 0]),
        odometry_noise_sim=np.array([0.05, 0.05]),
        measurement_noise_sim=np.array([0.08, 0.08])
    )
    
    simulator = RobotSimulatorR2(config)
    sim_data = simulator.simulate()
    
    # Build factor graph
    gfg = build_linear_factor_graph(
        sim_data,
        prior_noise_fg=np.array([0.05, 0.05]),
        odometry_noise_fg=np.array([0.1, 0.1]),
        measurement_noise_fg=np.array([0.1, 0.1])
    )
    
    # Solve
    solution = gfg.optimize()
    
    # Define variable keys
    pose_keys = [X(i+1) for i in range(len(sim_data['ground_truth']['poses']))]
    landmark_keys = [L(i+1) for i in range(len(sim_data['ground_truth']['landmarks']))]
    
    # Create analyzer
    analyzer = CovarianceAnalyzer(gfg, solution)
    
    # Generate report
    report = create_analysis_report(analyzer, pose_keys, landmark_keys)
    print(report)
    
    # Create visualizations
    visualizer = SLAMVisualizer(solution, sim_data['ground_truth'])
    fig = visualizer.plot_2d_slam(analyzer, pose_keys, landmark_keys, 
                                   show_correlation=True)
    plt.show()
    
    return analyzer, visualizer


if __name__ == "__main__":
    analyzer, visualizer = run_covariance_analysis_example()