from locTrack.basic_finctions import *
from mapGen import *

def motion_map_samples(pos, n, m, v, mu_a, sigma_a, K, building):
    samples = sample_generation_vectors(mu_a, sigma_a, K)
    motion_prob_map = np.zeros((n, m))
    acc_map = np.zeros((n, m, 2)) #n*m*2 = (n,m) * (a_x, a_y)
    for i in range(0, len(samples[0][0])):
        a = samples[0][i]
        p_a = samples[1][i]
        pos_t = np.around(pos + v * dt + a*dt*dt / 2)
        if (pos_t < np.array([n , m])).all() & (pos_t >= np.array([0 , 0])).all():
        #if not intersect_with_wall(pt(pos[0], pos[1]), pt(pos_t[0], pos_t[1]), building):
            motion_prob_map[pos_t[0]][pos_t[1]] = p_a
            acc_map[pos_t[0]][pos_t[1]][0] = a[0]
            acc_map[pos_t[0]][pos_t[1]][1] = a[1]

    return motion_prob_map, acc_map

def motion_map_distribution(pos, n, m, v, sigma_a, building, boosting = 1e100):
    prev = pos
    distribution = stats.multivariate_normal(mean=pos+v*dt, cov=[[sigma_a*dt ** 4 / 4,0],[0,sigma_a*dt ** 4 /4]])
    motion_prob_map = np.zeros((n, m))
    acc_map = np.zeros((n, m, 2))
    sum = 0
    for i in range(max(int(pos[0]) - 5*sigma_a, 0), min(int(pos[0]) + 5*sigma_a, n)):
        for j in range(max(int(pos[1]) - 5*sigma_a, 0), min(int(pos[1]) + 5*sigma_a, m)):
            #if not intersect_with_wall(pt(prev[0], prev[1]), pt(i, j), building):
                a = 2*(np.array([i , j]) - pos - v*dt) / dt ** 2
                acc_map[i][j][0] = a[0]
                acc_map[i][j][1] = a[1]
                motion_prob_map[i][j] = distribution.pdf([i, j])
                sum += motion_prob_map[i][j]

    boosting_coefficient = 0
    if sum > 0:
        boosting_coefficient = boosting / sum
    motion_prob_map = motion_prob_map * boosting_coefficient

    return motion_prob_map, acc_map


def motion_map_rotation(pos, n, m, v, sigma_a, building, boosting = 1e100): #point mobility model formula 7 -11?
    prev = pos
    distribution = stats.multivariate_normal(mean=pos+v*dt, cov=[[sigma_a*dt ** 4 / 4,0],[0,sigma_a*dt ** 4 /4]])   #не понимаю для чего это делаем? зачем среднее и ковариация

    motion_prob_map = np.zeros((n, m))
    acc_map = np.zeros((n, m, 2))
    sum = 0
    for i in range(max(int(pos[0]) - 5*sigma_a, 0), min(int(pos[0]) + 5*sigma_a, n)): # зачем такой интервал?
        for j in range(max(int(pos[1]) - 5*sigma_a, 0), min(int(pos[1]) + 5*sigma_a, m)): #зачем такой интервал?
            #if not intersect_with_wall(pt(prev[0], prev[1]), pt(i, j), building):
                a = 2*(np.array([i , j]) - pos - v*dt) / dt ** 2    #ускорение для х,у
                acc_map[i][j][0] = a[0] #записываем ускорение для каждой ячейке по Х
                acc_map[i][j][1] = a[1]#записываем ускорение для каждой ячейке по У

                mod_a = np.sqrt(a.dot(a))
                mod_v = np.sqrt(v.dot(v))
                scalar = np.dot(a, v)
                ratio = 0
                if (mod_v * mod_a > 0):
                    ratio = scalar / (mod_a * mod_v)
                phi = math.acos(min(1,max(ratio,-1))) # Что это такое? Угол?
                angle_prob = stats.norm(0, 0.5).pdf(phi) #вероятность угла поворота?

                motion_prob_map[i][j] = distribution.pdf([i, j]) * angle_prob #это что?
                sum += motion_prob_map[i][j]

    boosting_coefficient = 0
    if sum > 0:
        boosting_coefficient = boosting / sum       #for what
    motion_prob_map = motion_prob_map * boosting_coefficient

    return motion_prob_map, acc_map

def Viterbi(RSSI, env, path, pos, v): #RSSI - полученный уровень сигнала при мониторинге

    n = env.n
    m = env.m
    mu = env.mu #  Предварительная информация о вероятности ячеек p (k)
    sigma = env.sigma
    building = env.building
    # RSSI? -   [(array([-60.15052058]), 0),
    #           (array([-60.53896601]), 0),
    #           (array([-61.26909894]), 0),
    #           (array([-63.62395154]), 0),
    #           (array([-66.95994317]), 0), (array([-68.87480364]), 0), (array([-71.320633]), 0)]
    # env.mu?   [[-53.         -55.90730039 -58.28273777 -60.29114146 -62.03089987
                #  -63.56547554 -64.93820026 -66.17998081 -67.31363764 -68.35650083
                #  -69.32204133 -70.22093803 -71.06179974 -71.8516679  -72.59637541]
                # [-55.90730039 -56.94426173 -58.78552548 -60.58973486
    # env n,m =  15 X 15
    # path - real path - [(2, 2), (3, 1), (2, 3), (1, 5), (3, 7), (5, 8), (7, 10)]
    # pos - [2,2]
    # v - [0,0]
    path_est = []
    step = []
    #step.append(np.ndarray(shape = (3,n,m)))
    step.append(np.zeros((3, n, m))) #2*n*m : (0 = p, 1,2 = (v_x, v_y), (x,y)))
    step[0][0][pos[0]][pos[1]] = 1 #шаг на каждой итерации
    step[0][1][pos[0]][pos[1]] = v[0] #скорость по Х    по i итерации
    step[0][2][pos[0]][pos[1]] = v[1] #скорочть по У    по i итерации
    backward = [] #n*m*2: ((x, y), backward = (_x, _y))
    print("step = ", step)
    #forward step
    count = 0
    for el in RSSI[1:]:
        rssi = el[0][0]
        AP = el[1]
        prev_step = step[-1] #не понятно как и что значит
        curr_step = np.zeros((3, n, m)) #np.ndarray(shape = (3,n,m))
        curr_step_backward = np.zeros((n, m, 2)) #np.ndarray(shape = (n,m,2))
        rssi_prob = cond_prob(rssi, mu[AP], sigma) #получаем условную вероятность для каждого RSSI в строке?
        #### connect motion matrices
        for i in range(0, n):
            for j in range(0, m):
                m_p, a = motion_map_rotation(np.array([i, j]), n, m, np.array([prev_step[1][i][j], prev_step[2][i][j]]), sigma_a, building) #[[  7.75345215e+99   1.04931564e+99   2.60099343e+96   1.18084919e+92
    # 9.81910016e+85   0.00000000e+00   0.00000000e+00   0.00000000e+00
    # 0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
    # 0.00000000e+00   0.00000000e+00   0.000000    вероятность движения в ячейку? и ускорение?
                trans_prob = combine_prob(m_p, rssi_prob) #для каждой ячейки   point(4)ThesisProgress
                temp = np.full((n,m), prev_step[0][i][j])
                trans_prob = np.multiply(trans_prob, temp) #point(5)ThesisProgress Leave particles with max probability with some threshold?

                res = np.nonzero(trans_prob)
                for k in range(0, len(res[0])):
                    x = res[0][k]
                    y = res[1][k]
                    if curr_step[0][x][y] < trans_prob[x][y]: #зачем это делаем? по анализу 0 < 3.46214310292e-221
                        curr_step[0][x][y] = trans_prob[x][y]
                        curr_step_backward[x][y][0] = i
                        curr_step_backward[x][y][1] = j
                        acc = a[x][y]
                        curr_step[1][x][y] = prev_step[1][i][j] + acc[0] * dt
                        curr_step[2][x][y] = prev_step[2][i][j] + acc[1] * dt
        # max of curr_step 7.86570715293e+99 it's real ? curr_step -Вероятность появления и coordinate
        step.append(curr_step)  # что такое step
        backward.append(curr_step_backward)
        print("count = " , count)
        count += 1

    #backward step
    last_step = step[-1]
    best_samples = np.where(last_step[0] == last_step[0].max())     # что это такое?
    x = best_samples[0][0]
    y = best_samples[1][0]
    path_est.append((x, y))
    for i in range(len(backward)-1, -1, -1):
        _x = backward[i][x][y][0]
        _y = backward[i][x][y][1]
        path_est.append((_x, _y))
        x = int(_x)
        y = int(_y)

    path_est = path_est[::-1]

    print("Path estimation:")
    print(path_est)
    error(path, path_est)

    mu = np.zeros((n, m))
    for map in env.mu:
        mu = mu + map

    plot(mu, path, path_est)