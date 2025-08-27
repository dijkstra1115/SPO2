import numpy as np

def getsixchannels(rgb):

    yiq_i =  0.595716 * rgb[0, :] - 0.274453 * rgb[1, :] - 0.321263 * rgb[2, :] 
    ycgcr_cg = 128.0 -  81.085 * rgb[0, :] / 255.0 + 112.000 * rgb[1, :] / 255.0 - 30.915 * rgb[2, :] / 255.0
    ycgcr_cr = 128.0 + 112.000 * rgb[0, :] / 255.0 -  93.786 * rgb[1, :] / 255.0 - 18.214 * rgb[2, :] / 255.0
    ydbdr_dr = -1.333 * rgb[0, :] + 1.116 * rgb[1, :] + 0.217 * rgb[2, :] 
    pos_y = -2*rgb[0, :] + rgb[1, :] + rgb[2, :]
    chrom_x =   3*rgb[0, :] - 2*rgb[1, :]

    ret = [ycgcr_cg, ycgcr_cr, yiq_i, ydbdr_dr, pos_y, chrom_x]

    return ret

def getPOS(r_buf, g_buf, b_buf, win_len=30):
    epsilon = 1e-8
    
    if type(r_buf) is not np.ndarray or type(g_buf) is not np.ndarray or type(b_buf) is not np.ndarray:
        print('Inputs should be numpy arrays')
        ret = False
        ret_POS = np.array([])
        
    else:               
        if r_buf.size == 0 or g_buf.size == 0 or b_buf.size == 0:
            print("Empty inputs")
            ret = False
            ret_POS = np.array([])
        else:           
            ret = True
            ret_POS = np.zeros((1, r_buf.size))
            
            C = np.vstack((r_buf, g_buf, b_buf))
            
            color_base = np.array([[2, -1, -1], [0, -1, 1]])
            
            for i in range(r_buf.size - win_len + 1):
                c_tmp = C[:, i:i+win_len]
                mean_c = np.mean(c_tmp, axis=1)
                
                # if all channels' mean is close to 0, exit the window
                if np.allclose(mean_c, 0):
                    print("Zero color mean")
                    ret = False
                    ret_POS = np.array([])
                    break
                else:
                    c_tmp = c_tmp / (mean_c[:, None] + epsilon)
                    
                    s = color_base @ c_tmp   
                    
                    # when calculating the standard deviation of s, avoid division by 0
                    std_s0 = s[0, :].std()
                    std_s1 = s[1, :].std()
                    if np.isclose(std_s1, 0):
                        std_s1 = epsilon
                    alpha_base = np.array([[1, std_s0 / std_s1]])
                                        
                    p = alpha_base @ s   
                    
                    mean_p = np.mean(p)
                    std_p = np.std(p)
                    if np.isclose(std_p, 0):
                        std_p = epsilon
                        
                    ret_POS[0, i:i+win_len] = ret_POS[0, i:i+win_len] + ((p - mean_p) / std_p) 
    
    return ret, ret_POS