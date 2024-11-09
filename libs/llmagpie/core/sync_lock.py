from multiprocessing import Manager

manager = Manager()
store_ = manager.dict()


# import threading
# lock = threading.Lock()
