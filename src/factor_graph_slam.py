from __future__ import annotations

import gtsam
import numpy as np
from gtsam.symbol_shorthand import L, X

from association import Associator
from config import InferenceConfig
from models.dynamicmodels import OdometrySE2
from models.measurementmodels import RangeBearing
from result import StepRecord
from utils.utils_gtsam import (
    pose2_to_array,
    reorder_covariance_naive,
)
from utils.utils_math import make_psd
from utils.utils_victoria_park import Car, odometry_func, odom_increment_and_jac_from_ve_alpha
from landmark_manager import TentativeLandmarkManager, TentativeLandmark
from data_loader import SLAMStep



class FactorGraphSLAM:
    """Main SLAM estimator using factor graph."""

    def __init__(
        self,
        cfg: InferenceConfig,
        initial_pose: np.ndarray,
    ):  
        self.cfg = cfg
        self.associator = Associator(cfg)

        # Graph and values
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        
        # ISAM2 stuff 
        self.new_factors = gtsam.NonlinearFactorGraph()
        self.new_values = gtsam.Values()
        isam_params = gtsam.ISAM2Params()
        self.isam = gtsam.ISAM2(isam_params)

        self.tentative_manager = TentativeLandmarkManager(
            M = self.cfg.M, 
            N = self.cfg.N, 
            association_gate = self.cfg.association_gate
        )

        # Models
        self.motion_model = OdometrySE2( # TODO: remove
            sigma_x=cfg.noise.x_std,
            sigma_y=cfg.noise.y_std,
            sigma_theta=cfg.noise.theta_std_rad,
        )
        self.sensor_model = RangeBearing(
            sigma_range=cfg.noise.range_std, 
            sigma_bearing=cfg.noise.bearing_std_rad,
            sensor_offset=np.array(cfg.sensor_offset),
        )

        # Noise models
        self.odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(cfg.noise.odometry_std)
        self.measurement_noise = gtsam.noiseModel.Diagonal.Sigmas(cfg.noise.measurement_std)

        # State tracking
        self.num_poses = 0 
        self.num_landmarks = 0 

        # Initialize graph with prior on initial pose
        initial_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(cfg.noise.prior_std)
        self._add_prior_factor(gtsam.Pose2(*initial_pose), initial_pose_noise)

        # Odometry integration
        self.Delta = gtsam.Pose2()
        self.Sigma = np.zeros((3,3))
        self.poses_dr = [gtsam.Pose2(*initial_pose)]  # for dead reckoning trajectory


    def _add_prior_factor(self, prior_pose: gtsam.Pose2, prior_pose_noise: np.ndarray):
        """Add prior factor for initial pose."""
 
        prior_factor = gtsam.PriorFactorPose2(X(0), prior_pose, prior_pose_noise)
        
        self.graph.add(prior_factor)
        self.values.insert(X(0), prior_pose)

        self.new_factors.add(prior_factor)
        self.new_values.insert(X(0), prior_pose)
        self.num_poses += 1 


    def get_predicted_measurements(self, pose_pred: gtsam.Pose2) -> tuple[np.ndarray, np.ndarray]:
        """Get predicted measurements for all landmarks based on priori pose estimate and landmark estimates."""
        
        M = self.num_landmarks

        zbar = np.zeros((M,2), dtype=float)  
        zbar_ids = np.zeros(M, dtype=int) 
        
        for j in range(self.num_landmarks):
            lm = self.values.atPoint2(L(j))
            zbar[j,0] = pose_pred.range(lm)
            zbar[j,1] = pose_pred.bearing(lm).theta() 
            zbar_ids[j] = j 
        return zbar, zbar_ids
    
    def gate_predicted_measurements(self, zbar, zbar_ids) -> tuple[np.ndarray, np.ndarray]:
        """Gate predicted measurements based on range and bearing thresholds."""
        zbar_gated = []
        zbar_gated_ids = []
        for z, id in zip(zbar, zbar_ids):
            r = z[0] # range
            b = z[1] # bearing
            if r < self.cfg.range_gate and np.abs(b) < np.deg2rad(self.cfg.fov_gate_deg)/2:
                zbar_gated.append(z)
                zbar_gated_ids.append(id)  

        return np.array(zbar_gated, dtype=float).reshape(-1, 2), np.array(zbar_gated_ids, dtype=int)
    
    def extract_information(self, zbar_ids: np.ndarray) -> np.ndarray:
        pose_key = X(self.num_poses - 1)
        keys = [pose_key] + [L(i) for i in zbar_ids]

        marginals = gtsam.Marginals(self.graph, self.values)
        Lambda = marginals.jointMarginalInformation(keys).fullMatrix()
        
        Lambda = reorder_covariance_naive(Lambda) # TODO: is this correct
        
        return Lambda

    def compute_innovation_covariance_from_info(self, zbar_ids, pose_pred, Lambda):
        n = len(zbar_ids)
        H = np.zeros((2*n, 3 + 2*n))
        R = np.zeros((2*n, 2*n))
        R_i = np.diag((self.cfg.noise.range_std**2,
                    self.cfg.noise.bearing_std_rad**2))

        for i, lid in enumerate(zbar_ids):
            m_i = self.values.atPoint2(L(lid))
            H_x  = self.sensor_model.H_x(pose2_to_array(pose_pred), m_i)
            H_mi = self.sensor_model.H_m(pose2_to_array(pose_pred), m_i)
            H[2*i:2*i+2, 0:3] = H_x
            H[2*i:2*i+2, 3+2*i:3+2*i+2] = H_mi
            R[2*i:2*i+2, 2*i:2*i+2] = R_i

        # Solve Lambda * V = H^T  ->  V = Lambda^{-1} H^T
        V = np.linalg.solve(Lambda, H.T)          # (3+2n) x (2n)
        S = H @ V + R                              # (2n) x (2n)

        # numeric hygiene
        S = 0.5 * (S + S.T)
        S = make_psd(S)
        return S
    
    def info_func(self, marginals, keys):
        return marginals.jointMarginalInformation(keys).fullMatrix()
    
    def inverse_func(self, matrix):
        return np.linalg.inv(matrix)
    
    def get_marginals_(self) -> gtsam.Marginals:
        return gtsam.Marginals(self.graph, self.values)

    def extract_covariance(self, zbar_ids: np.ndarray) -> np.ndarray:
        """Extract joint covariance for last pose and predicted measurements coresponding to the ids in zbar_ids."""
        
        if len(zbar_ids) == 0:
            print("No predicted measurements after gating, TODO")
        
        pose_pred_key = X(self.num_poses-1)
        
        keys = [pose_pred_key]  # NOTE: order in which the keys are added is important
        keys += [L(id) for id in zbar_ids]   

        self.optimize_graph()  # ensure values are up to date before extracting covariance
        # marginals = gtsam.Marginals(self.isam.getFactorsUnsafe(), self.values)
        # marginals = gtsam.Marginals(self.graph, self.values)
        # marginals = self.get_marginals_()

        # Joint covariance for previous pose and local landmarks (in local frame)
        # covariance  = marginals.jointMarginalCovariance(keys).fullMatrix()
        # information = self.info_func(marginals, keys)
        # covariance = self.inverse_func(information)

        covariance = self.isam.jointMarginalCovariance(keys)
        # information = marginals.jointMarginalInformation(keys).fullMatrix()
        # covariance = np.linalg.inv(information)
        # print(covariance.shape)

        # Reorder covariance to match state ordering
        covariance = reorder_covariance_naive(covariance) # TODO: maybe make more secure
        # covariance  = reorder_covariance_auto(
        #     covariance ,
        #     source_keys=sorted(keys),
        #     target_keys=keys,
        #     values=self.values,
        # )

        # lfg = self.graph.linearize(self.values)
        # Lambda = lfg.hessian()[0]
        # Cov = np.linalg.inv(Lambda)

        # marginals = gtsam.Marginals(self.graph, self.values)

        # pose_key = X(self.num_poses-1)
        # landmark_keys = [L(id) for id in zbar_ids]

        # n = len(landmark_keys)
        # covariance = np.zeros((3 + 2 * n, 3 + 2 * n))  # (pose + landmarks, pose + landmarks)
        # for i, lm_key in enumerate(landmark_keys):
        #     cov = marginals.jointMarginalCovariance([pose_key, lm_key]).fullMatrix() # 5x5 covariance for pose and landmark i
        #     cov_xx = cov[0:3, 0:3]  # covariance of pose
        #     cov_xm = cov[0:3, 3:5]  # cross-covariance between pose and landmark i
        #     cov_mm = cov[3:5, 3:5]  # covariance of landmark i

        #     covariance[:3,:3] = cov_xx
        #     covariance[:3, 3 + 2 * i : 3 + 2 * i + 2] = cov_xm
        #     covariance[3 + 2 * i : 3 + 2 * i + 2, :3] = cov_xm.T
        #     covariance[3 + 2 * i : 3 + 2 * i + 2, 3 + 2 * i : 3 + 2 * i + 2] = cov_mm

        return covariance


    def compute_innovation_covariance(
        self,
        zbar_ids: np.ndarray,
        pose_pred: gtsam.Pose2,
        cov_body: np.ndarray,
    ):
        """Compute innovation covariance for predicted measurements."""
        
        n = len(zbar_ids) # num predicted measurements

        H = np.zeros((2 * n, 3 + 2 * n))
        R = np.zeros((2 * n, 2 * n))
        
        R_i = np.diag((self.cfg.noise.range_std**2, self.cfg.noise.bearing_std_rad**2))
 
        for i, id in enumerate(zbar_ids):
            m_i = self.values.atPoint2(L(id))
            H_x = self.sensor_model.H_x(pose2_to_array(pose_pred), m_i)
            H_mi = self.sensor_model.H_m(pose2_to_array(pose_pred), m_i)
            H[2 * i : 2 * i + 2, 0:3] = H_x
            H[2 * i : 2 * i + 2, 3 + 2 * i : 3 + 2 * i + 2] = H_mi
            R[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = R_i

        S = H @ cov_body @ H.T + R
        S = make_psd(S) 

        return S

    def compute_association(
            self, 
            z: np.ndarray, 
            zbar: np.ndarray, 
            zbar_ids: np.ndarray, 
            S: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]: 
        """Compute association between measurements and predicted measurements using self.associator."""
        
        # Do association
        association_indices = self.associator.associate(z, zbar, S)
    
        # As Assoicator.associate returns indices into zbar for each measurement, we need to convert to landmark IDs
        association_ids = association_indices.copy()  
        association_ids[association_indices >= 0] = [zbar_ids[idx] for idx in association_indices if idx >= 0]

        return association_ids, association_indices
    

    
    # def process_step(self, z_odometry: tuple[float, float, float], z_range_bearing: list[tuple[float, float]]) -> StepRecord:
    # def process_step(self, odometry: gtsam.Pose2, measurements: np.ndarray) -> StepRecord:
    def process_step(self, data: SLAMStep) -> StepRecord:
        """Main SLAM step processing:

        Parameters
        -------
        z_odometry : TODO np.ndarray (3,)
            (u, v, psi) representing odometry measurement (relative motion)
        z_range_bearing: np.ndarray (M,2)
            (range, bearing) measurements to landmarks

        Returns
        -------
        StepRecord 
            current estimates, measurements, associations etc. for this step

        """
        
        odometry, J_odo_u = odom_increment_and_jac_from_ve_alpha(data.ve_dr, data.alpha_dr, data.dt_dr)

        # odo = gtsam.Pose2(*odometry)  
        H1 = np.zeros((3,3), order='F')
        H2 = np.zeros((3,3), order='F')
    
        self.Delta = self.Delta.compose(odometry, H1, H2)

        Q_u = np.diag(np.array([0.1, 0.5*np.pi/180])**2)
        Q_f = np.diag(self.cfg.noise.odometry_std**2)
        
        Q_odo = J_odo_u @ Q_u @ J_odo_u.T + Q_f
        # Q_odo = Q_f

        self.Sigma = H1 @ self.Sigma @ H1.T + H2 @ Q_odo @ H2.T

        # pose_pred = self._add_odometry(odo)

        measurements = data.measurements
        
        if len(measurements) == 0:
            return None
        else:
            from_idx = self.num_poses - 1
            to_idx = self.num_poses

            odom_factor = gtsam.BetweenFactorPose2(
                X(from_idx), X(to_idx), self.Delta, gtsam.noiseModel.Gaussian.Covariance(self.Sigma)
            )
            self.graph.add(odom_factor)
            self.new_factors.add(odom_factor)

            # Update dead reckoning trajectory
            self.poses_dr.append(self.poses_dr[-1].compose(self.Delta))

            # Predict next pose for initialization
            pose_prev = self.values.atPose2(X(from_idx))
            pose_pred = pose_prev.compose(self.Delta)
            self.values.insert(X(to_idx), pose_pred)
            self.new_values.insert(X(to_idx), pose_pred)
            self.num_poses += 1 # pose added

            # Reset accumulated odom
            self.Delta = gtsam.Pose2()
            self.Sigma = np.zeros((3,3))

            # Data assocation
            zbar, zbar_ids = self.get_predicted_measurements(pose_pred)
            zbar_gated, zbar_gated_ids = self.gate_predicted_measurements(zbar, zbar_ids)
            
            cov = self.extract_covariance(zbar_gated_ids)
            
            cov_innovation = self.compute_innovation_covariance(zbar_gated_ids, pose_pred, cov)
            # Lambda = self.extract_information(zbar_gated_ids)
            # cov_innovation = self.compute_innovation_covariance_from_info(zbar_gated_ids, pose_pred, Lambda)
            asssoc_ids, assoc_idx = self.compute_association(measurements, zbar_gated, zbar_gated_ids, cov_innovation)
                
            self._add_associated_landmark_measurements(measurements, asssoc_ids)
            confirmed_tentatives = self._process_unassociated_measurements(measurements, asssoc_ids)
            self._promote_tentative_landmarks(confirmed_tentatives)
            # self._add_landmark_measurements(measurements, asssoc_ids)
            
            self.optimize_graph()


            # ---- store history ----
            record = StepRecord(
                step=self.num_poses-1,
                poses=self.get_estimated_poses(), 
                # poses_cov=self.get_estimated_pose_covariances(),
                poses_dr=self.get_poses_dr(),
                landmarks=self.get_estimated_landmarks(), 
                # landmarks_cov=self.get_estimated_landmark_covariances(),
                measurements=measurements,
                predicted_measurements=zbar_gated,
                predicted_measurements_ids=zbar_gated_ids,
                associations_ids=asssoc_ids,
                associations_idx=assoc_idx,
                cov_innovation=cov_innovation,
            )

            return record


    def optimize_graph(self) -> gtsam.Values:
        """Run optimization on the current factor graph"""
        if self.cfg.algorithm == "isam2":
            self.isam.update(self.new_factors, self.new_values)
            self.new_factors = gtsam.NonlinearFactorGraph()
            self.new_values = gtsam.Values()
            self.values = self.isam.calculateEstimate()
        elif self.cfg.algorithm == "batch":  # full batch optimization
            optParams = gtsam.LevenbergMarquardtParams()
            optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.values, optParams)
            self.values = optimizer.optimize()
        else:
            raise ValueError(f"Unknown algorithm: {self.cfg.algorithm}")

    # def _add_odometry(self, odometry: gtsam.Pose2) -> gtsam.Pose2:
    #     """Add odometry factor between consecutive poses"""
    #     from_idx = self.num_poses - 1
    #     to_idx = self.num_poses

    #     odom_factor = gtsam.BetweenFactorPose2(
    #         X(from_idx), X(to_idx), odometry, self.odometry_noise
    #     )
    #     self.graph.add(odom_factor)
    #     self.new_factors.add(odom_factor)

    #     # Predict next pose for initialization
    #     pose_prev = self.values.atPose2(X(from_idx))
    #     pose_pred = pose_prev.compose(odometry)
    #     self.values.insert(X(to_idx), pose_pred)
    #     self.new_values.insert(X(to_idx), pose_pred)
        
    #     self.num_poses += 1 # pose added
        
    #     return pose_pred
    
    def _add_associated_landmark_measurements(
        self, 
        measurements: np.ndarray, 
        associations: np.ndarray
    ):
        """Add factors only for measurements associated with confirmed landmarks."""
        pose_key = X(self.num_poses-1)

        for (r, b), a_j in zip(measurements, associations):
            if a_j >= 0:  # measurement j associated with previously observed landmark a_j
                meas_factor = gtsam.BearingRangeFactor2D(
                    pose_key, L(a_j), gtsam.Rot2(b), r, self.measurement_noise
                )
                self.graph.add(meas_factor)
                self.new_factors.add(meas_factor)
            elif a_j in (-1, -2):  
                # -1: unassociated -> handled by tentative manager
                # -2: ambiguous/outlier -> currently ignore
                continue
            else: 
                raise ValueError(f"Invalid association index: {a_j}")
            
    def _process_unassociated_measurements(
        self,
        measurements: np.ndarray,   # (M, 2), columns = [range, bearing]
        associations: np.ndarray,   # (M,)
    ) -> list:
        """
        Send unassociated measurements to tentative landmark manager.

        Returns a list of tentative landmarks that are now confirmed and ready
        to be promoted into the factor graph.
        """
        pose_key = X(self.num_poses - 1)
        current_pose = self.values.atPose2(pose_key)

        world_measurements = []
        raw_measurements = []

        for (r, b), a_j in zip(measurements, associations):
            if a_j == -1:
                lm_x_local = r * np.cos(b)
                lm_y_local = r * np.sin(b)
                lm_local = gtsam.Point2(lm_x_local, lm_y_local)
                lm_global = current_pose.transformFrom(lm_local)

                world_measurements.append(np.array([lm_global[0], lm_global[1]]))
                raw_measurements.append(np.array([r, b]))

        confirmed_tentatives = self.tentative_manager.process_unassociated_measurements(
            current_step=self.num_poses - 1,
            world_measurements=world_measurements,
            raw_measurement=raw_measurements,
        )

        return confirmed_tentatives
    
    def _promote_tentative_landmarks(self, confirmed_tentatives: list[TentativeLandmark]) -> None:
        """
        Promote confirmed tentative landmarks into the factor graph.

        Simple version:
        - insert landmark variable
        - initialize its position
        - add factor to the pose of the most recent supporting observation
        """
        for tlm in confirmed_tentatives:
            lm_id = self.num_landmarks
            lm_key = L(lm_id)
            self.num_landmarks += 1

            lm_global = gtsam.Point2(float(tlm.position[0]), float(tlm.position[1]))
            self.values.insert(lm_key, lm_global)
            self.new_values.insert(lm_key, lm_global)

            # Can adjust to retroactively add old measurement factors
            # obs = tlm.supporting_observations[-1]
            # r, b = obs.measurement

            # meas_factor = gtsam.BearingRangeFactor2D(
            #     pose_key,
            #     lm_key,
            #     gtsam.Rot2(float(b)),
            #     float(r),
            #     self.measurement_noise,
            # )
            # self.graph.add(meas_factor)
            # self.new_factors.add(meas_factor)

            for obs in tlm.supporting_observations:
                r, b = obs.measurement

                meas_factor = gtsam.BearingRangeFactor2D(
                    X(obs.step),
                    lm_key,
                    gtsam.Rot2(float(b)),
                    float(r),
                    self.measurement_noise,
                )
                self.graph.add(meas_factor)
                self.new_factors.add(meas_factor)


            # Optional:
            # store mapping if you want to remember which tentative became which landmark
            # self.confirmed_landmark_metadata[lm_id] = ...


    # def _add_landmark_measurements(
    #     self, 
    #     measurements: np.ndarray, # (M,2)
    #     associations: np.ndarray, # (M,) 
    # ):
    #     """Add landmark measurement factors"""
    #     pose_key = X(self.num_poses-1)

    #     # j is measurement index, a_j is associated landmark index
    #     for (r, b), a_j in zip(measurements, associations):
    #         if a_j >= 0:  # measurement j associated with previously observed landmark a_j
    #             meas_factor = gtsam.BearingRangeFactor2D(pose_key, L(a_j), gtsam.Rot2(b), r, self.measurement_noise)
    #             self.graph.add(meas_factor)
    #             self.new_factors.add(meas_factor)
    #         elif a_j == -1:  # new landmark, initialize and add factor
    #             lm_key = L(self.num_landmarks)
    #             self.num_landmarks += 1
    #             meas_factor = gtsam.BearingRangeFactor2D(pose_key, lm_key, gtsam.Rot2(b), r, self.measurement_noise)
    #             self.graph.add(meas_factor)
    #             self.new_factors.add(meas_factor)
    #             self._initialize_landmark(lm_key, pose_key, r, b)
    #         # elif a_j == -2:  # ambiguous association, could be outlier or valid match. For now, treat as outlier (no factor)
    #         #     continue
    #         else: 
    #             raise ValueError(f"Invalid association index: {a_j}")
        
    # def _initialize_landmark(self, lm_key: int, pose_key: int, range: float, bearing: float) -> None:
    #     """Initialize a newly observed landmark."""
    #     current_pose = self.values.atPose2(pose_key)

    #     # Convert from polar to Cartesian in robot frame
    #     lm_x_local = range * np.cos(bearing)
    #     lm_y_local = range * np.sin(bearing)
        
    #     # Transform from local to global frame
    #     lm_local = gtsam.Point2(lm_x_local, lm_y_local)
    #     lm_global = current_pose.transformFrom(lm_local)
        
    #     # Insert into values
    #     self.values.insert(lm_key, lm_global)
    #     self.new_values.insert(lm_key, lm_global)


    def get_marginals(self) -> gtsam.Marginals:
        """Compute marginals for current estimate"""
        return gtsam.Marginals(self.graph, self.values)

    def get_current_pose(self) -> gtsam.Pose2:
        """Get current robot pose estimate"""
        return self.values.atPose2(X(self.num_poses-1))
    
    def get_estimated_poses(self) -> np.ndarray:
        """Get all pose estimates up to current step"""
        return np.array([pose2_to_array(self.values.atPose2(X(k))) for k in range(self.num_poses)])

    def get_estimated_pose_covariances(self) -> list[np.ndarray]:
        """Get covariances for all pose estimates"""
        marginals = self.get_marginals()
        return [marginals.marginalCovariance(X(k)) for k in range(self.num_poses)]

    def get_estimated_landmarks(self) -> np.ndarray:
        """Get all landmark estimates up to current step"""
        return np.array([self.values.atPoint2(L(lm)) for lm in range(self.num_landmarks)])

    def get_estimated_landmark_covariances(self) -> list[np.ndarray]:
        """Get covariances for all landmark estimates"""
        marginals = self.get_marginals()
        return [marginals.marginalCovariance(L(lm)) for lm in range(self.num_landmarks)]

    def get_poses_dr(self) -> np.ndarray:
        """Get dead reckoning trajectory"""
        return np.array([pose2_to_array(pose) for pose in self.poses_dr])