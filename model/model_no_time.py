from model import get_data, train_no_time

if __name__ == "__main__":
	x_train, y_train, x_test, y_test = get_data()
	train_no_time(x_train, y_train, x_test, y_test)