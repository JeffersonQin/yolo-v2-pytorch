from typing import Optional
from matplotlib import pyplot as plt
from . import globalvar as G


def draw_precision_recall(pr_data: list, class_idx: Optional[int]=None):
	"""Draw Precision-Recall Curve

	Args:
		pr_data (list): Precision Recall Curve Data
		class_idx (Optional[int]): Class index, used to render title
	"""
	p = [data['precision'] for data in pr_data]
	r = [data['recall'] for data in pr_data]

	plt.plot(r, p, 'o-', color='r')
	plt.xlabel("Recall")
	plt.ylabel("Precision")

	if class_idx is not None:
		plt.title(G.get('categories')[class_idx])

	plt.show()
