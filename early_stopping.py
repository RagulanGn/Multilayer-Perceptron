
class EarlyStopping:
	def __init__(self, patience=5, min_delta=0.0, mode='max'):
		self.patience = patience
		self.min_delta = min_delta
		self.mode = mode

		self.factor = 1 if mode == 'max' else -1
		self.patience_count = 0
		self.EarlyStop = False

		self.best_score = None
		self.best_params = None

	#ignite.handlers.early_stopping.EarlyStopping(patience, score, min_delta=0.0, cumulative_delta=False, min_delta_mode='abs', mode='max'
	def __call__(self, current_score, params):
		if self.best_score is None :
			self.best_score = current_score
			self.best_params = params
		elif self.factor * self.best_score + self.min_delta < self.factor * current_score:
			self.best_score = current_score
			self.best_params = params
			self.patience_count = 0
		else :
			self.patience_count += 1
			if self.patience_count >= self.patience:
				self.EarlyStop = True
		return self.EarlyStop

	def get_best_params(self):
		return self.best_params
