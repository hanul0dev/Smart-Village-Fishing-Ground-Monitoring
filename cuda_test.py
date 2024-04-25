class Metric(SimpleClass):
    """
    Class for computing evaluation metrics for YOLOv8 model.

    Attributes:
        p (list): Precision for each class. Shape: (nc,).
        r (list): Recall for each class. Shape: (nc,).
        f1 (list): F1 score for each class. Shape: (nc,).
        all_ap (list): AP scores for all classes and all IoU thresholds. Shape: (nc, 10).
        ap_class_index (list): Index of class for each AP score. Shape: (nc,).
        nc (int): Number of classes.

    Methods:
        ap50(): AP at IoU threshold of 0.5 for all classes. Returns: List of AP scores. Shape: (nc,) or [].
        ap(): AP at IoU thresholds from 0.5 to 0.95 for all classes. Returns: List of AP scores. Shape: (nc,) or [].
        mp(): Mean precision of all classes. Returns: Float.
        mr(): Mean recall of all classes. Returns: Float.
        map50(): Mean AP at IoU threshold of 0.5 for all classes. Returns: Float.
        map75(): Mean AP at IoU threshold of 0.75 for all classes. Returns: Float.
        map(): Mean AP at IoU thresholds from 0.5 to 0.95 for all classes. Returns: Float.
        mean_results(): Mean of results, returns mp, mr, map50, map.
        class_result(i): Class-aware result, returns p[i], r[i], ap50[i], ap[i].
        maps(): mAP of each class. Returns: Array of mAP scores, shape: (nc,).
        fitness(): Model fitness as a weighted combination of metrics. Returns: Float.
        update(results): Update metric attributes with new evaluation results.
    """

    def __init__(self) -> None:
        """Initializes a Metric instance for computing evaluation metrics for the YOLOv8 model."""
        self.p = []  # (nc, )
        self.r = []  # (nc, )
        self.f1 = []  # (nc, )
        self.all_ap = []  # (nc, 10)
        self.ap_class_index = []  # (nc, )
        self.nc = 0

    @plt_settings()
    def plot_pr_curve(px, py, ap, save_dir=Path('./yolov8m/beachSmall/pr_curve.png'), names=(), on_plot=None):
        """Plots a precision-recall curve."""
        fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
        py = np.stack(py, axis=1)

        if 0 < len(names) < 21:  # display per-class legend if < 21 classes
            for i, y in enumerate(py.T):
                ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
        else:
            ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

        ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
        ax.set_title('Precision-Recall Curve')
        fig.savefig(save_dir, dpi=250)
        plt.close(fig)
        if on_plot:
            on_plot(save_dir)