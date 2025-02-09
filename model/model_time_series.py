from model import get_data, train_time_series

if __name__ == "__main__":
	x_train, y_train, x_test, y_test = get_data()
	train_time_series(x_train, y_train, x_test, y_test)