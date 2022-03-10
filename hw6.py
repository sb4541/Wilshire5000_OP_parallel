import numpy as np
import pickle as pkl
import time
from multiprocessing import Process, Lock, Value
from multiprocessing.sharedctypes import RawArray
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#If the status of the worker is 0, then it means that it is asleep (no task to be done).
#If the status.value of the worker is bigger than 0 (), then it means that it is working or about to start working on
#pairs captured by [(status.value-1)*step:(status.value)*step].
#If the status of the worker is -1, then it means that it is quitting.

def ave_return_calculations(status, commonlock, speciallock, raw_sum_returns_stocks, raw_ave_returns_pairs, raw_stocks_1, raw_stocks_2, raw_number_steps, raw_number_times, raw_step, raw_number_pairs):
    commonlock.acquire()
    num_steps = raw_number_steps.value
    num_times = raw_number_times.value - 1
    num_pairs = raw_number_pairs.value
    sum_returns_stocks = np.array(raw_sum_returns_stocks)
    commonlock.release()

    flag = 1
    while flag != 0:
        speciallock.acquire()
        localstatus = status.value
        speciallock.release()
        if localstatus == -1:
            flag = 0
        elif localstatus > 0:
            if localstatus != num_steps:
                commonlock.acquire()
                localvalues1 = sum_returns_stocks[raw_stocks_1[(localstatus - 1) * raw_step.value: localstatus * raw_step.value]]
                localvalues2 = sum_returns_stocks[raw_stocks_2[(localstatus - 1) * raw_step.value: localstatus * raw_step.value]]
                commonlock.release()
                localvalue = (localvalues1 - localvalues2) / num_times
                raw_ave_returns_pairs[:] = localvalue
            else:
                commonlock.acquire()
                localvalues1 = sum_returns_stocks[raw_stocks_1[localstatus * raw_step.value: ]]
                localvalues2 = sum_returns_stocks[raw_stocks_2[localstatus * raw_step.value: ]]
                commonlock.release()
                localvalue = (localvalues1 - localvalues2) / num_times
                commonlock.acquire()
                raw_ave_returns_pairs[: num_pairs - num_steps * raw_step.value] = localvalue
                commonlock.release()
            speciallock.acquire()
            status.value = 0
            speciallock.release()
        else:
            time.sleep(0.5)



def coefficients_calculations(status, commonlock, commonlock_coeff, speciallock, raw_batch_size, raw_coefficients, raw_x_old, raw_factor, raw_number_steps, raw_step, extended_raw_ave_returns_pairs, raw_number_pairs):
    commonlock.acquire()
    num_pairs = raw_number_pairs.value
    num_steps = raw_number_steps.value
    step = raw_step.value
    batch_size = raw_batch_size.value
    commonlock.release()

    flag = 1
    while flag != 0:
        speciallock.acquire()
        localstatus = status.value
        speciallock.release()
        if localstatus == -1:
            flag = 0
        elif localstatus > 0:
            if localstatus != num_steps:
                localx_old = np.array(raw_x_old)
                localfactor = np.array(raw_factor).reshape((batch_size, step))
                localave_returns_pairs = np.array(extended_raw_ave_returns_pairs[(localstatus - 1) * step: localstatus * step])
                localvalue = np.dot(localfactor - localave_returns_pairs, localx_old)
                commonlock_coeff.acquire()
                coefficients = np.array(raw_coefficients)
                coefficients += localvalue
                raw_coefficients[:] = coefficients
                commonlock_coeff.release()
            else:
                localx_old = np.array(raw_x_old[: num_pairs - num_steps * step])
                localfactor = np.array(raw_factor[: batch_size * (num_pairs - num_steps * step)]).reshape((batch_size, num_pairs - num_steps * step))
                localave_returns_pairs = np.array(extended_raw_ave_returns_pairs[localstatus * step: ])
                localvalue = np.dot(localfactor - localave_returns_pairs, localx_old)
                commonlock_coeff.acquire()
                coefficients = np.array(raw_coefficients)
                coefficients += localvalue
                raw_coefficients[:] = coefficients
                commonlock_coeff.release()
            speciallock.acquire()
            status.value = 0
            speciallock.release()
        else:
            time.sleep(0.5)

def stoch_gradient_calculations(status, commonlock, speciallock, raw_batch_size, raw_coefficients, raw_factor, raw_number_steps, raw_step, extended_raw_ave_returns_pairs, raw_number_pairs, raw_grad, theta):
    commonlock.acquire()
    num_pairs = raw_number_pairs.value
    num_steps = raw_number_steps.value
    step = raw_step.value
    batch_size = raw_batch_size.value
    coefficients = np.array(raw_coefficients)
    commonlock.release()

    flag = 1
    while flag != 0:
        speciallock.acquire()
        localstatus = status.value
        speciallock.release()
        if localstatus == -1:
            flag = 0
        elif localstatus > 0:
            if localstatus != num_steps:
                localfactor = np.array(raw_factor).reshape((batch_size, step))
                localave_returns_pairs = np.array(extended_raw_ave_returns_pairs[(localstatus - 1) * step: localstatus * step])
                raw_grad[(localstatus - 1) * step: localstatus * step] = 2 * theta * np.sum( np.transpose( localfactor - localave_returns_pairs ) * coefficients, axis=1)/(batch_size-1) - localave_returns_pairs
            else:
                localfactor = np.array(raw_factor[: batch_size * (num_pairs - num_steps * step)]).reshape((batch_size, num_pairs - num_steps * step))
                localave_returns_pairs = np.array(extended_raw_ave_returns_pairs[localstatus * step: ])
                raw_grad[num_steps * step: ] = 2 * theta * np.sum( np.transpose( localfactor - localave_returns_pairs ) * coefficients, axis=1)/(batch_size-1) - localave_returns_pairs
            speciallock.acquire()
            status.value = 0
            speciallock.release()
        else:
            time.sleep(0.5)



def step_calculations(status, commonlock, speciallock, raw_number_steps, raw_step, raw_number_pairs, raw_grad, raw_x_old, raw_min_length, raw_delta):
    commonlock.acquire()
    num_pairs = raw_number_pairs.value
    num_steps = raw_number_steps.value
    step = raw_step.value
    commonlock.release()

    flag = 1
    while flag != 0:
        speciallock.acquire()
        localstatus = status.value
        speciallock.release()
        if localstatus == -1:
            flag = 0
        elif localstatus > 0:
            if localstatus != num_steps:
                localx_old = np.array(raw_x_old)
                localubound = np.array(localx_old > 1 - 1e-14, dtype=int)
                locallbound = np.array(localx_old < -1 + 1e-14, dtype=int)
                localgrad = np.array(raw_grad[(localstatus - 1) * step: localstatus * step])
                localcheck_grad = np.array(localgrad < -1e-14, dtype=int)
                localvalue = -localgrad + localgrad * localubound * localcheck_grad + localgrad * locallbound * (np.ones(step) - localcheck_grad)
                raw_delta[:] = localvalue
                del localcheck_grad
                if raw_min_length.value == -1.0:
                    raw_min_length.value = np.min(np.abs((np.sign(localvalue) - localx_old) / localvalue))
                elif np.min(np.abs((np.sign(localvalue) - localx_old) / localvalue)) < raw_min_length.value:
                    raw_min_length.value = np.min(np.abs((np.sign(localvalue) - localx_old) / localvalue))
            else:
                localx_old = np.array(raw_x_old[: num_pairs - num_steps * step])
                localubound = np.array(localx_old > 1 - 1e-14, dtype=int)
                locallbound = np.array(localx_old < -1 + 1e-14, dtype=int)
                localgrad = np.array(raw_grad[localstatus * step: ])
                localcheck_grad = np.array(localgrad < -1e-14, dtype=int)
                localvalue = -localgrad + localgrad * localubound * localcheck_grad + localgrad * locallbound * (np.ones(len(localcheck_grad)) - localcheck_grad)
                raw_delta[: num_pairs - num_steps * step] = localvalue
                del localcheck_grad
                if raw_min_length.value == -1:
                    raw_min_length.value = np.min(np.abs((np.sign(localvalue) - localx_old) / localvalue))
                elif np.min(np.abs((np.sign(localvalue) - localx_old) / localvalue)) < raw_min_length.value:
                    raw_min_length.value = np.min(np.abs((np.sign(localvalue) - localx_old) / localvalue))

            speciallock.acquire()
            status.value = 0
            speciallock.release()
        else:
            time.sleep(0.5)




if __name__ == '__main__':

#############
        # EXTRACT DATA
    with open('w5000.bin', 'rb') as f:
        data = pkl.load(f)
    data = data[1:, 3:]

#############
    # GET RID OF INVALID COLUMNS (ASSETS)
    store = []
    for i in range(3234):
        for j in range(2769):
            if data[j, i] == "":
                store.append(i)
                break
    print('Number of assets without some data: ' + str(len(store)))
    print('Number of assets with all the data: ' + str(3234 - len(store)))
    clean_data = np.delete(data, store, axis=1)
    clean_data = clean_data.astype(float)
    number_stocks = clean_data.shape[1]
    number_times = clean_data.shape[0]
    print('Number of assets: ' + str(number_stocks))
    print('Number of days: ' + str(number_times))

#############
    # CREATE PAIRS
    stocks_1 = []
    stocks_2 = []
    for i in range(number_stocks):
        for j in range(number_stocks):
            if i < j:
                stocks_1.append(i)
                stocks_2.append(j)

#############
    # FIND STOCKS RETURNS AND INITIALIZE RETURNS OF PAIRS
    returns_stocks = np.diff(clean_data, axis=0) / clean_data[:-1]
    number_pairs = len(stocks_1)
    extended_raw_ave_returns_pairs = RawArray('d', np.zeros(number_pairs))

#############
    # GET RID OF HEAVY UNUSED STRUCTURES
    del data
    del clean_data

#############
    # MULTIPROCESSING FOR AVERAGE RETURNS CALCULATIONS
    step = 30000
    number_steps = 75
    #step * number_steps is almost number_pairs

    raw_step = Value('i', step)
    raw_number_steps = Value('i', number_steps)
    raw_number_times = Value('i', number_times)
    raw_number_pairs = Value('i', number_pairs)
    sum_returns_stocks = np.sum(returns_stocks, axis=0)
    raw_sum_returns_stocks = RawArray('d', sum_returns_stocks)
    del sum_returns_stocks
    raw_stocks_1 = RawArray('i', stocks_1)
    raw_stocks_2 = RawArray('i', stocks_2)
    del stocks_1
    del stocks_2
    number_workers = 2
    commonlock = Lock()
    process = {}
    status = {}
    speciallock = {}
    raw_ave_returns_pairs = {}
    tasks = np.zeros(number_workers, dtype='int')
    t1 = time.time()
    for worker in range(1, number_workers + 1):
        status[worker] = Value('i', worker)
        speciallock[worker] = Lock()
        raw_ave_returns_pairs[worker] = RawArray('d', step)
        process[worker] = Process(target=ave_return_calculations, args=(status[worker], commonlock, speciallock[worker], raw_sum_returns_stocks, raw_ave_returns_pairs[worker], raw_stocks_1, raw_stocks_2, raw_number_steps, raw_number_times, raw_step, raw_number_pairs))
        process[worker].start()
        tasks[worker - 1] = worker - 1
    remainingjobs = number_steps-number_workers
    while remainingjobs > 0:
        for worker in range(1, number_workers+1):
            speciallock[worker].acquire()
            wstatus = status[worker].value
            speciallock[worker].release()
            if wstatus == 0 and remainingjobs > 0:  # not working
                extended_raw_ave_returns_pairs[tasks[worker - 1] * step: (tasks[worker - 1] + 1) * step] = raw_ave_returns_pairs[worker]
                speciallock[worker].acquire()
                status[worker].value = number_steps - remainingjobs + 1
                tasks[worker - 1] = status[worker].value - 1
                speciallock[worker].release()
                remainingjobs -= 1
    time.sleep(1)
    for worker in range(1, number_workers + 1):
        terminate_worker = False
        task = tasks[worker - 1]
        while not terminate_worker:
            speciallock[worker].acquire()
            wstatus = status[worker].value
            speciallock[worker].release()
            if wstatus == 0:
                if task != number_steps:
                    extended_raw_ave_returns_pairs[tasks[worker - 1] * step: (tasks[worker - 1] + 1) * step] = raw_ave_returns_pairs[worker]
                else:
                    extended_raw_ave_returns_pairs[tasks[worker - 1] * step:] = raw_ave_returns_pairs[worker][number_steps * step:]
                speciallock[worker].acquire()
                status[worker].value = -1
                speciallock[worker].release()
                terminate_worker = True
            else:
                time.sleep(1)
    for worker in range(1, number_workers + 1):
        process[worker].join()
    t2 = time.time()
    print('Number of pairs is', number_pairs)
    t_avg = t2 - t1
    print('Time to find the average return for each pair is: ' + str(t2-t1))
    print('Time to find the average return for each pair without multiprocessing (HW5) was 57.58693528175354')
    print('First values of average pair returns are: ' + str(extended_raw_ave_returns_pairs[: 8]))
    del raw_ave_returns_pairs
    del raw_sum_returns_stocks
    stocks_1 = np.array(raw_stocks_1)
    del raw_stocks_1
    stocks_2 = np.array(raw_stocks_2)
    del raw_stocks_2


    print('Time to find the stochastic gradient is about 84')
    print('Time to find the stochastic gradient in HW5 was around 154')
#############
    # MULTIPROCESSING FOR STOCHASTIC GRADIENT DESCENT
    ave_returns_pairs = np.array(extended_raw_ave_returns_pairs)
    x_old = np.array(ave_returns_pairs > 0, dtype=int) - np.array(ave_returns_pairs < 0, dtype=int)
    x_new = x_old

    returns = []
    volatility = []
    for theta in [0.01, 0.025, 0.06, 0.1]:
        x_old = np.array(ave_returns_pairs > 0, dtype=int) - np.array(ave_returns_pairs < 0, dtype=int)
        x_new = x_old
        delta = np.ones(number_pairs)
        raw_grad = RawArray('d', number_pairs)
        tolerance = 1e-09
        mpc_flag = 1
        grad = np.array(raw_grad)

        batch_size = 200
        raw_batch_size = Value('i', batch_size)
        raw_coefficients = RawArray('d', np.zeros(raw_batch_size.value))
        iteration = 0
        counter = 10
        min_length = 1.0
        raw_x_old = {}
        t1_desc = time.time()
        while min_length * np.linalg.norm(delta) > tolerance and iteration < 20:
            commonlock_coeff = Lock()
            if counter == 10:
                # MULTIPROCESSING FOR 'coefficients'
                t1 = time.time()
                batch = np.random.choice(number_times - 1, size=raw_batch_size.value, replace=False)
                raw_factor = {}
                for worker in range(1, number_workers + 1):
                    status[worker].value = worker
                    raw_x_old[worker] = RawArray('d', x_old[(worker - 1) * step: worker * step])
                    raw_factor[worker] = RawArray('d', (
                                returns_stocks[:, stocks_1[(worker - 1) * step: worker * step]][batch, :] - returns_stocks[:,
                                                                                                            stocks_2[(
                                                                                                                                 worker - 1) * step: worker * step]][
                                                                                                            batch,
                                                                                                            :]).flatten())
                    process[worker] = Process(target=coefficients_calculations, args=(
                    status[worker], commonlock, commonlock_coeff, speciallock[worker], raw_batch_size, raw_coefficients,
                    raw_x_old[worker], raw_factor[worker], raw_number_steps, raw_step, extended_raw_ave_returns_pairs,
                    raw_number_pairs))
                    process[worker].start()
                remainingjobs = number_steps - number_workers
                while remainingjobs > 0:
                    for worker in range(1, number_workers + 1):
                        speciallock[worker].acquire()
                        wstatus = status[worker].value
                        speciallock[worker].release()
                        if wstatus == 0 and remainingjobs > 0:  # not working
                            if remainingjobs != 1:
                                speciallock[worker].acquire()
                                status[worker].value = number_steps - remainingjobs + 1
                                raw_x_old[worker][:] = x_old[(status[worker].value - 1) * step: status[worker].value * step]
                                commonlock.acquire()
                                raw_factor[worker][:] = (returns_stocks[:, stocks_1[(status[worker].value - 1) * step: status[
                                                                                                                           worker].value * step]][
                                                         batch, :] - returns_stocks[:, stocks_2[
                                                                                       (status[worker].value - 1) * step:status[
                                                                                                                             worker].value * step]][
                                                                     batch, :]).flatten()
                                commonlock.release()
                                speciallock[worker].release()
                            else:
                                speciallock[worker].acquire()
                                status[worker].value = number_steps - remainingjobs + 1
                                raw_x_old[worker][: number_pairs - number_steps * step] = x_old[status[worker].value * step:]
                                commonlock.acquire()
                                raw_factor[worker][: batch_size * (number_pairs - number_steps * step)] = (
                                            returns_stocks[:, stocks_1[status[worker].value * step:]][batch,
                                            :] - returns_stocks[:, stocks_2[status[worker].value * step:]][batch, :]).flatten()
                                commonlock.release()
                                speciallock[worker].release()
                            remainingjobs -= 1
                time.sleep(1)
                for worker in range(1, number_workers + 1):
                    terminate_worker = False
                    while not terminate_worker:
                        speciallock[worker].acquire()
                        wstatus = status[worker].value
                        speciallock[worker].release()
                        if wstatus == 0:
                            speciallock[worker].acquire()
                            status[worker].value = -1
                            speciallock[worker].release()
                            terminate_worker = True
                        else:
                            time.sleep(1)
                for worker in range(1, number_workers + 1):
                    process[worker].join()
                del commonlock_coeff
                # print('Vector of coefficients has been found\n')

                # MULTIPROCESSING FOR STOCHASTIC GRADIENT
                for worker in range(1, number_workers + 1):
                    status[worker].value = worker
                    raw_factor[worker] = RawArray('d', (
                                returns_stocks[:, stocks_1[(worker - 1) * step: worker * step]][batch, :] - returns_stocks[:,
                                                                                                            stocks_2[(
                                                                                                                                 worker - 1) * step: worker * step]][
                                                                                                            batch,
                                                                                                            :]).flatten())
                    process[worker] = Process(target=stoch_gradient_calculations, args=(
                    status[worker], commonlock, speciallock[worker], raw_batch_size, raw_coefficients, raw_factor[worker],
                    raw_number_steps, raw_step, extended_raw_ave_returns_pairs, raw_number_pairs, raw_grad, theta))
                    process[worker].start()
                remainingjobs = number_steps - number_workers
                while remainingjobs > 0:
                    for worker in range(1, number_workers + 1):
                        speciallock[worker].acquire()
                        wstatus = status[worker].value
                        speciallock[worker].release()
                        if wstatus == 0 and remainingjobs > 0:  # not working
                            if remainingjobs != 1:
                                speciallock[worker].acquire()
                                status[worker].value = number_steps - remainingjobs + 1
                                commonlock.acquire()
                                raw_factor[worker][:] = (returns_stocks[:, stocks_1[(status[worker].value - 1) * step: status[
                                                                                                                           worker].value * step]][
                                                         batch, :] - returns_stocks[:, stocks_2[
                                                                                       (status[worker].value - 1) * step:status[
                                                                                                                             worker].value * step]][
                                                                     batch, :]).flatten()
                                commonlock.release()
                                speciallock[worker].release()
                            else:
                                speciallock[worker].acquire()
                                status[worker].value = number_steps - remainingjobs + 1
                                commonlock.acquire()
                                raw_factor[worker][: batch_size * (number_pairs - number_steps * step)] = (
                                            returns_stocks[:, stocks_1[status[worker].value * step:]][batch,
                                            :] - returns_stocks[:, stocks_2[status[worker].value * step:]][batch, :]).flatten()
                                commonlock.release()
                                speciallock[worker].release()
                            remainingjobs -= 1
                time.sleep(1)
                for worker in range(1, number_workers + 1):
                    terminate_worker = False
                    while not terminate_worker:
                        speciallock[worker].acquire()
                        wstatus = status[worker].value
                        speciallock[worker].release()
                        if wstatus == 0:
                            speciallock[worker].acquire()
                            status[worker].value = -1
                            speciallock[worker].release()
                            terminate_worker = True
                        else:
                            time.sleep(1)
                for worker in range(1, number_workers + 1):
                    process[worker].join()
                # print('Gradient has been found\n')
                t2 = time.time()
                # print('Time to find gradient is: ' + str(t2-t1))

                # print(raw_grad[: 10])
                counter = 0
            else:
                counter += 1

            if mpc_flag == 1 and counter == 0:
                grad = np.array(raw_grad)

            # MULTIPROCESSING FOR STOCHASTIC GRADIENT DESCENT STEP
            t1_delta = time.time()
            min_length = -1.0
            if mpc_flag == 1:
                for i in range(number_steps):
                    ubound = np.array(x_old[i * step:(i + 1) * step] > 1 - 1e-14, dtype=int)
                    lbound = np.array(x_old[i * step:(i + 1) * step] < -1 + 1e-14, dtype=int)
                    check_grad = np.array(grad[i * step:(i + 1) * step] < -1e-14, dtype=int)
                    delta[i * step:(i + 1) * step] = -grad[i * step:(i + 1) * step] + grad[i * step:(
                                                                                                                i + 1) * step] * ubound * check_grad + grad[
                                                                                                                                                       i * step: (
                                                                                                                                                                             i + 1) * step] * lbound * (
                                                                 np.ones(step) - check_grad)
                    if min_length == -1:
                        min_length = np.min(np.abs(
                            (np.sign(delta[i * step:(i + 1) * step]) - x_old[i * step:(i + 1) * step]) / delta[i * step:(
                                                                                                                                    i + 1) * step]))
                    if np.min(np.abs((np.sign(delta[i * step:(i + 1) * step]) - x_old[i * step:(i + 1) * step]) / delta[
                                                                                                                  i * step:(
                                                                                                                                   i + 1) * step])) < min_length:
                        min_length = np.min(np.abs(
                            (np.sign(delta[i * step:(i + 1) * step]) - x_old[i * step:(i + 1) * step]) / delta[i * step:(
                                                                                                                                    i + 1) * step]))
                ubound = np.array(x_old[number_steps * step:] > 1 - 1e-14, dtype=int)
                lbound = np.array(x_old[number_steps * step:] < -1 + 1e-14, dtype=int)
                check_grad = np.array(grad[number_steps * step:] < -1e-14, dtype=int)
                delta[number_steps * step:] = -grad[number_steps * step:] + grad[
                                                                            number_steps * step:] * ubound * check_grad + grad[
                                                                                                                          number_steps * step:] * lbound * (
                                                          np.ones(len(check_grad)) - check_grad)
                if np.min(np.abs((np.sign(delta[number_steps * step:]) - x_old[number_steps * step:]) / delta[
                                                                                                        number_steps * step:])) < min_length:
                    min_length = np.min(np.abs(
                        (np.sign(delta[number_steps * step:]) - x_old[number_steps * step:]) / delta[number_steps * step:]))

            else:
                raw_delta = {}
                raw_min_length = {}
                tasks = np.zeros(number_workers, dtype='int')
                for worker in range(1, number_workers + 1):
                    status[worker].value = worker
                    raw_delta[worker] = RawArray('d', np.ones(step))
                    tasks[worker - 1] = worker - 1
                    raw_min_length[worker] = Value('d', min_length)
                    raw_x_old[worker][:] = x_old[(worker - 1) * step: worker * step]
                    process[worker] = Process(target=step_calculations, args=(
                        status[worker], commonlock, speciallock[worker], raw_number_steps, raw_step, raw_number_pairs, raw_grad,
                        raw_x_old[worker], raw_min_length[worker], raw_delta[worker]))
                    process[worker].start()
                remainingjobs = number_steps - number_workers
                while remainingjobs > 0:
                    for worker in range(1, number_workers + 1):
                        speciallock[worker].acquire()
                        wstatus = status[worker].value
                        speciallock[worker].release()
                        if wstatus == 0 and remainingjobs > 0:  # not working
                            delta[tasks[worker - 1] * step: (tasks[worker - 1] + 1) * step] = np.array(raw_delta[worker])
                            if min_length == -1.0:
                                min_length = raw_min_length[worker].value
                            elif raw_min_length[worker].value < min_length:
                                min_length = raw_min_length[worker].value
                            if remainingjobs != 1:
                                speciallock[worker].acquire()
                                status[worker].value = number_steps - remainingjobs + 1
                                raw_x_old[worker][:] = x_old[(status[worker].value - 1) * step: status[worker].value * step]
                                tasks[worker - 1] = status[worker].value - 1
                                raw_min_length[worker].value = 1.0
                                speciallock[worker].release()
                            else:
                                speciallock[worker].acquire()
                                status[worker].value = number_steps - remainingjobs + 1
                                raw_x_old[worker][: number_pairs - number_steps * step] = x_old[status[worker].value * step:]
                                raw_min_length[worker].value = 1.0
                                tasks[worker - 1] = status[worker].value - 1
                                speciallock[worker].release()
                            remainingjobs -= 1
                time.sleep(0.1)
                for worker in range(1, number_workers + 1):
                    task = tasks[worker - 1]
                    terminate_worker = False
                    while not terminate_worker:
                        speciallock[worker].acquire()
                        wstatus = status[worker].value
                        speciallock[worker].release()
                        if wstatus == 0:
                            if min_length == -1.0:
                                min_length = raw_min_length[worker].value
                            elif raw_min_length[worker].value < min_length:
                                min_length = raw_min_length[worker].value
                            if task != number_steps:
                                delta[task * step: (task + 1) * step] = np.array(raw_delta[worker])
                            else:
                                delta[task * step:] = np.array(raw_delta[worker][: number_pairs - number_steps * step])
                            speciallock[worker].acquire()
                            status[worker].value = -1
                            speciallock[worker].release()
                            terminate_worker = True
                        else:
                            time.sleep(1)
                for worker in range(1, number_workers + 1):
                    process[worker].join()

            t2_delta = time.time()
            x_new = x_old + min_length * delta
            # print('Iteration', iteration)
            # print('Time to find delta was', t2_delta - t1_delta)
            # print('Difference between points', np.linalg.norm(x_new - x_old))
            x_old = x_new
            iteration += 1
            # print('Norm of step', np.linalg.norm(delta))
            # print('Step length', min_length)

        t2_desc = time.time()
        print('Point has been found, first values are: ' + str(x_new[: 8]))
        print('Number of iterations was ', iteration)
        print('Total running time was: ' + str(t2_desc - t1_desc + t_avg))
        print('Total running time in HW5 was around 360')

        #FIND THE RETURN
        # raw_x = {}
        # current_return = 0.0
        # for worker in range(1, worker + 1):
        #     status[worker].value = worker
        #     raw_x[worker] = RawArray('d', x_new[(worker - 1) * step: worker * step])
        #     raw_return[worker] = Value('d', step)
        #     process[worker] = Process(target=return_calculations, args=(status[worker], speciallock[worker], raw_number_steps, raw_step, raw_number_pairs, raw_x[worker], extended_raw_ave_returns_pairs))
        #     process[worker].start()
        # remainingjobs = number_steps - number_workers
        # while remainingjobs > 0:
        #     for worker in range(1, number_workers + 1):
        #         speciallock[worker].acquire()
        #         wstatus = status[worker].value
        #         speciallock[worker].release()
        #         if wstatus == 0 and remainingjobs > 0:  # not workingù
        #             current_return += np.array(raw_return[worker].value)
        #             if remainingjobs != 1:
        #                 speciallock[worker].acquire()
        #                 status[worker].value = number_steps - remainingjobs + 1
        #                 raw_x[worker][:] = x_new[(status[worker].value - 1) * step: status[worker].value * step]
        #                 speciallock[worker].release()
        #             else:
        #                 speciallock[worker].acquire()
        #                 status[worker].value = number_steps - remainingjobs + 1
        #                 raw_x[worker][: number_pairs - number_steps * step] = x_new[status[worker].value * step:]
        #                 speciallock[worker].release()
        #             remainingjobs -= 1
        # time.sleep(0.1)
        # for worker in range(1, number_workers + 1):
        #     terminate_worker = False
        #     while not terminate_worker:
        #         speciallock[worker].acquire()
        #         wstatus = status[worker].value
        #         speciallock[worker].release()
        #         if wstatus == 0:
        #             current_return += np.array(raw_return[worker].value)
        #             speciallock[worker].acquire()
        #             status[worker].value = -1
        #             speciallock[worker].release()
        #             terminate_worker = True
        #         else:
        #             time.sleep(1)
        # for worker in range(1, number_workers + 1):
        #     process[worker].join()
        current_return = 0.0
        for i in range(number_steps):
            current_return += np.dot(x_new[i * step: (i + 1) * step], np.array(extended_raw_ave_returns_pairs[i * step: (i + 1) * step]))
        current_return += np.dot(x_new[number_steps * step: ], np.array(extended_raw_ave_returns_pairs[number_steps * step: ]))
        returns.append(current_return)


        #FIND THE VOLATILITY
        current_volatility = 0.0
        for i in range(number_steps):
            current_volatility += np.linalg.norm(np.dot(returns_stocks[:, stocks_1[(i - 1) * step: i * step]] - returns_stocks[:, stocks_2[(i - 1) * step: i * step]] - np.array(extended_raw_ave_returns_pairs[i * step: (i + 1) * step]), x_new[i * step: (i + 1) * step])) ** 2
        current_volatility += np.linalg.norm(np.dot(returns_stocks[:, stocks_1[number_steps * step: ]] - returns_stocks[:,stocks_2[number_steps * step: ]] - np.array(extended_raw_ave_returns_pairs[number_steps * step: ]), x_new[number_steps * step: ])) ** 2
        volatility.append(current_volatility)




    plt.plot(volatility, returns)

