import numpy as np 
from scipy.spatial.transform import Rotation as R 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm 
import matplotlib.colors as mcolors
from tqdm import tqdm
from scipy import stats
import multiprocessing as mp 

# GEOMETRIA DO PROBLEMA
Raio_Datum_A = X.XX
Raio_apoios_turbina = X.XX 
Gap_nom = X.XX 
Tol_lin = {'Carcaça': X.XX, 'Distribuidor_turbina': X.XX, 'Disco': X.XX, 'Corpo_turbina': X.XX} 
Tol_geo = {'Carcaça': X.XX, 'Distribuidor_turbina': X.XX, 'Disco': X.XX} 
Pos_apoios_ang = [X.XX, X.XX, X.XX, X.XX] 
p_apoio_nom = [np.array([Raio_apoios_turbina * np.cos(np.radians(a)), Raio_apoios_turbina * np.sin(np.radians(a)), 0]) for a in Pos_apoios_ang]
ved_nom = X.XX
tol_ved = X.XX
Sigma_sup = X.XX

def fit_plane_and_get_rot(p_reais):
    pts = np.array(p_reais)
    centroid = np.mean(pts, axis=0)
    c_pts = pts - centroid
    _, _, vh = np.linalg.svd(c_pts)
    normal = vh[2, :]
    if normal[2] < 0: normal = -normal 
    z_axis = np.array([0, 0, 1])
    v = np.cross(z_axis, normal)
    c = np.dot(z_axis, normal)
    s = np.linalg.norm(v)
    if s == 0: return np.eye(3) 
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

# FUNÇÃO TRABALHADORA OTIMIZADA PARA RAM
def simular_bloco(n_it):
    res_comp = []
    gaps_bim = []
    desvios_max = []
    pior_rot_local = None
    maior_amp_local = -1
    contagem_falhas_angulares = np.zeros(360) # Contador para o mapa de risco
    rads = np.radians(np.arange(0, 360, 1))
    
    for _ in range(n_it):
        # Sorteios de Tolerâncias
        h_ved = np.random.normal(ved_nom, tol_ved/Sigma_sup)
        ez_c = np.random.normal(0, (Tol_lin['Carcaça'])/Sigma_sup, 4)
        ea_c = np.random.normal(0, np.degrees(np.arctan(Tol_geo['Carcaça']/(Raio_apoios_turbina*2)))/Sigma_sup)
        ez_td = np.random.normal(0, (Tol_lin['Distribuidor_turbina'])/Sigma_sup, 4)
        ea_td = np.random.normal(0, np.degrees(np.arctan(Tol_geo['Distribuidor_turbina']/(Raio_apoios_turbina*2)))/Sigma_sup)
        ez_d = np.random.normal(0, (Tol_lin['Disco'])/Sigma_sup)
        ea_d = np.random.normal(0, np.degrees(np.arctan(Tol_geo['Disco']/(Raio_Datum_A*2)))/Sigma_sup)

        # Cálculos de Rotação e Planos
        rot_c_apoio = fit_plane_and_get_rot([p + [0, 0, ez] for p, ez in zip(p_apoio_nom, ez_c)])
        rot_td_apoio = fit_plane_and_get_rot([p + [0, 0, ez] for p, ez in zip(p_apoio_nom, ez_td)])
        
        ang_c = np.random.uniform(0, 2*np.pi)
        ang_t = np.random.uniform(0, 2*np.pi)
        
        rot_c_paral = R.from_euler('xy', [ea_c*np.cos(ang_c), ea_c*np.sin(ang_c)], degrees=True).as_matrix()
        rot_td_prop = R.from_euler('xy', [ea_td*np.cos(ang_t), ea_td*np.sin(ang_t)], degrees=True).as_matrix()
        rot_d_prop = R.from_euler('xy', [ea_d*np.cos(ang_c), ea_d*np.sin(ang_c)], degrees=True).as_matrix()

        # Resultado da Vedação
        rot_total_ved = rot_d_prop @ rot_td_prop @ rot_td_apoio @ rot_c_paral @ rot_c_apoio
        z_vars_ved = rot_total_ved[2,0]*Raio_Datum_A*np.cos(rads) + rot_total_ved[2,1]*Raio_Datum_A*np.sin(rads)
        
        amp_inc = np.max(np.abs(z_vars_ved))
        
        # Guardar a pior rotação para o relatório
        if amp_inc > maior_amp_local:
            maior_amp_local = amp_inc
            pior_rot_local = rot_total_ved

        # Cálculo do Gap Real
        translacao_central = np.mean(ez_c) + np.mean(ez_td) + ez_d
        gaps_da_peca = Gap_nom + translacao_central + z_vars_ved
        
        # ACUMULADOR DE FALHAS: Verifica vazamento em cada grau (0-359)
        falhas_nesta_peca = (gaps_da_peca - h_ved) > 0
        contagem_falhas_angulares += falhas_nesta_peca.astype(int)
        
        # Resultados para Histogramas
        res_comp.append(np.max(gaps_da_peca - h_ved))
        gaps_bim.extend([Gap_nom + amp_inc, Gap_nom - amp_inc])
        desvios_max.append(amp_inc)
        
    return res_comp, gaps_bim, desvios_max, pior_rot_local, contagem_falhas_angulares

if __name__ == '__main__':
    iteracoes = 10000000
    num_cores = max(1, mp.cpu_count() - 2)
    
    num_tarefas = 1000 
    it_por_tarefa = iteracoes // num_tarefas
    tarefas = [it_por_tarefa] * num_tarefas

    print(f"Iniciando simulação paralela em {num_cores} núcleos...")
    
    resultados_brutos = []
    with mp.Pool(num_cores) as pool:
        for res in tqdm(pool.imap_unordered(simular_bloco, tarefas), 
                        total=num_tarefas, 
                        desc=f"Simulando {iteracoes:,} Peças"):
            resultados_brutos.append(res)

    resultados_compressao = [item for sub in resultados_brutos for item in sub[0]]
    gaps_extremos_bimodal = [item for sub in resultados_brutos for item in sub[1]]
    desvios_maximos_por_peca = [item for sub in resultados_brutos for item in sub[2]]
    
    # Encontra a pior rotação global entre as retornadas pelos blocos
    piores_rots = [sub[3] for sub in resultados_brutos if sub[3] is not None]
    rads_ref = np.radians(np.arange(0, 360, 1))
    amps = [np.max(np.abs(r[2,0]*Raio_Datum_A*np.cos(rads_ref) + r[2,1]*Raio_Datum_A*np.sin(rads_ref))) for r in piores_rots]
    pior_rot_global = piores_rots[np.argmax(amps)]

    # --- SEÇÃO DE ANÁLISE (PARETO) ---
    def get_rms_sens():
        delta = 0.01
        g_ref = np.radians(np.arange(0, 360, 1))
        px, py = Raio_Datum_A * np.cos(g_ref), Raio_Datum_A * np.sin(g_ref)
        def calc_resp(ez_c, ez_t, ez_d, ea_c, ea_t, ea_d, h_v):
            rc = fit_plane_and_get_rot([p + [0, 0, ez] for p, ez in zip(p_apoio_nom, ez_c)])
            rt = fit_plane_and_get_rot([p + [0, 0, ez] for p, ez in zip(p_apoio_nom, ez_t)])
            r_par_c = R.from_euler('x', ea_c, degrees=True).as_matrix()
            r_par_t = R.from_euler('x', ea_t, degrees=True).as_matrix()
            r_par_d = R.from_euler('x', ea_d, degrees=True).as_matrix()
            rot = r_par_d @ r_par_t @ rt @ r_par_c @ rc
            resps = Gap_nom + (rot[2,0]*px + rot[2,1]*py + np.mean(ez_c) + np.mean(ez_t) + ez_d) - h_v
            return resps[np.argmax(np.abs(resps))]
        base = calc_resp([0]*4, [0]*4, 0, 0, 0, 0, 5.2)
        sens = {}
        for i in range(4):
            v = [0]*4; v[i] = delta
            sens[f'Ap{i+1}_Carc'] = np.abs(calc_resp(v, [0]*4, 0, 0, 0, 0, 5.2) - base) / delta
            sens[f'Ap{i+1}_Turb'] = np.abs(calc_resp([0]*4, v, 0, 0, 0, 0, 5.2) - base) / delta
        sens['Par_Carc'] = np.abs(calc_resp([0]*4, [0]*4, 0, delta, 0, 0, 5.2) - base) / delta
        sens['Par_Turb'] = np.abs(calc_resp([0]*4, [0]*4, 0, 0, delta, 0, 5.2) - base) / delta
        sens['Par_Disc'] = np.abs(calc_resp([0]*4, [0]*4, 0, 0, 0, delta, 5.2) - base) / delta
        sens['Lin_Disc'] = 1.0; sens['Vedacao'] = 1.0
        return sens

    S = get_rms_sens()
    sigmas_in = {**{f'Ap{i+1}_Carc': (Tol_lin['Carcaça']*2)/Sigma_sup for i in range(4)},
                 **{f'Ap{i+1}_Turb': (Tol_lin['Distribuidor_turbina']*2)/Sigma_sup for i in range(4)},
                 'Par_Carc': np.degrees(np.arctan(Tol_geo['Carcaça']/(Raio_apoios_turbina*2)))/Sigma_sup,
                 'Par_Turb': np.degrees(np.arctan(Tol_geo['Distribuidor_turbina']/(Raio_apoios_turbina*2)))/Sigma_sup,
                 'Par_Disc': np.degrees(np.arctan(Tol_geo['Disco']/(Raio_Datum_A*2)))/Sigma_sup,
                 'Lin_Disc': (Tol_lin['Disco']*2)/Sigma_sup,
                 'Vedacao': (tol_ved*2)/Sigma_sup}
    
    var_ind = {k: (S[k] * sigmas_in[k])**2 for k in S.keys()}
    total_v = sum(var_ind.values())
    sorted_contrib = dict(sorted({k: (v/total_v)*100 for k, v in var_ind.items()}.items(), key=lambda x: x[1], reverse=True))

    # --- VISUALIZAÇÃO ---
    plt.figure(figsize=(10, 5)); plt.hist(gaps_extremos_bimodal, bins=100, color='skyblue', edgecolor='darkblue', alpha=0.7)
    plt.axvline(Gap_nom, color='red', label=f'Nominal ({Gap_nom}mm)'); plt.title("Gap distribution"); plt.legend(); plt.show()

    plt.figure(figsize=(10, 5)); plt.hist(resultados_compressao, bins=50, color='yellow', edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linewidth=3, label='Limite de Vazamento'); plt.title("Sealing compression histogram"); plt.legend(); plt.show()

    # --- VISUALIZAÇÃO PARETO (CORRIGIDO) ---
    plt.figure(figsize=(10, 7))
    labels = list(sorted_contrib.keys())
    valores = list(sorted_contrib.values())
    
    plt.barh(labels[::-1], valores[::-1], color='lightgreen', edgecolor='black')
    
    # ESTA LINHA EVITA QUE OS NÚMEROS VAZEM:
    # Dá uma folga de 15% além da maior barra
    plt.xlim(0, max(valores) * 1.15) 
    
    for i, v in enumerate(valores[::-1]): 
        plt.text(v + 0.5, i, f'{v:.1f}%', va='center', fontweight='bold')
        
    plt.title("Tolerances contribution")
    plt.tight_layout()
    plt.show()
    
    # --- PROCESSAMENTO DOS DADOS PARA O MAPA ---
    # Soma os arrays de falhas de todos os núcleos (índice 4 do retorno)
    total_falhas_por_grau = np.sum([sub[4] for sub in resultados_brutos], axis=0)
    probabilidade_vazamento_ppm = (total_falhas_por_grau / iteracoes) * 1000000.

    # --- GERAÇÃO DO MAPA DE CALOR (RISCO POR ÂNGULO) ---
    print("Gerando Mapa de Risco por Probabilidade Angular...")
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='polar')
    
    # Define a malha: 360 graus e uma faixa para a largura da vedação
    theta_plot = np.radians(np.arange(0, 360, 1))
    r_faixa = np.array([Raio_Datum_A - 5, Raio_Datum_A + 5])
    
    # Cria a matriz de coordenadas e a matriz de cores (2, 360)
    theta_m, r_m = np.meshgrid(theta_plot, r_faixa)
    z_m = np.tile(probabilidade_vazamento_ppm, (2, 1))
    
    # Plotagem usando o mapa 'YlOrRd' (Amarelo para Vermelho) para indicar risco
    quadmesh = ax.pcolormesh(theta_m, r_m, z_m, cmap='YlOrRd', shading='gouraud')
    
    # Configurações estéticas do gráfico polar
    ax.set_yticklabels([]) 
    ax.set_theta_zero_location('E') # Coloca o 0° na Direita (Leste)
    ax.set_theta_direction(1)      # Sentido Anti-Horário (Positivo)
    ax.set_rlim(0, Raio_Datum_A + 15)
    
    # Barra de cores lateral
    cbar = fig.colorbar(quadmesh, pad=0.1, shrink=0.7)
    cbar.set_label('Risco de Vazamento Local [PPM]')
    
    plt.title(f"Failure map: probability of failure according to the angle"
              f"(Sampling: {iteracoes:,} peças)", 
              fontsize=13, fontweight='bold', pad=20)
    
    # Adiciona a marcação dos apoios para referência visual
    for ang, i in zip(Pos_apoios_ang, range(4)):
        ax.annotate(f'Ap.{i+1}', xy=(np.radians(ang), Raio_apoios_turbina), 
                    xytext=(np.radians(ang), Raio_apoios_turbina + 10),
                    arrowprops=dict(facecolor='black', arrowstyle='->', alpha=0.5), 
                    fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.show()
    
    # --- CÁLCULO DA PIOR PEÇA (INDISPENSÁVEL PARA O GRÁFICO E RELATÓRIO) ---
    piores_candidatas = [sub[3] for sub in resultados_brutos if sub[3] is not None]
    rads_ref = np.radians(np.arange(0, 360, 1))
    
    # Identifica a rotação com maior desvio absoluto
    amplitudes = [np.max(np.abs(r[2,0]*Raio_Datum_A*np.cos(rads_ref) + \
                                r[2,1]*Raio_Datum_A*np.sin(rads_ref))) for r in piores_candidatas]
    
    pior_rot_global = piores_candidatas[np.argmax(amplitudes)]
    
    # DEFINIÇÃO DA VARIÁVEL:
    pior_desvio_3D = pior_rot_global[2,0]*Raio_Datum_A*np.cos(rads_ref) + \
                     pior_rot_global[2,1]*Raio_Datum_A*np.sin(rads_ref)
    
    # --- MAPA TÉRMICO DO PIOR CASO (VISUALIZAÇÃO COMPLEMENTAR) ---
    print("Gerando Mapa Térmico do Pior Caso Simulado...")
    fig_pior = plt.figure(figsize=(9, 7))
    ax_pior = fig_pior.add_subplot(111, projection='polar')
    
    # Reutiliza a malha e o pior_desvio_3D calculado para o relatório
    z_pior_m = np.tile(pior_desvio_3D, (2, 1))
    
    # Escala simétrica para o Turbo (Azul frio / Vermelho quente)
    v_limit = np.max(np.abs(pior_desvio_3D))
    quadmesh_pior = ax_pior.pcolormesh(theta_m, r_m, z_pior_m, 
                                       cmap='turbo', vmin=-v_limit, vmax=v_limit, shading='gouraud')
    
    ax_pior.set_yticklabels([]); ax_pior.set_theta_zero_location('E') 
    ax_pior.set_theta_direction(1)
    ax_pior.set_rlim(0, Raio_Datum_A + 15)
    
    cbar_pior = fig_pior.colorbar(quadmesh_pior, pad=0.1, shrink=0.7)
    cbar_pior.set_label('Desvio do Plano [mm] (Ref. Nominal)')
    
    plt.title(f"Mapa Térmico: Pior Caso Individual (Amplitude Máxima)\n"
              f"Desvio Máximo Encontrado: ±{np.max(np.abs(pior_desvio_3D)):.3f} mm", 
              fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()
    
    piores_candidatas = [sub[3] for sub in resultados_brutos if sub[3] is not None]
    rads_ref = np.radians(np.arange(0, 360, 1))
    
    # 2. Identifica qual rotação causou a maior amplitude absoluta
    amplitudes = [np.max(np.abs(r[2,0]*Raio_Datum_A*np.cos(rads_ref) + \
                                r[2,1]*Raio_Datum_A*np.sin(rads_ref))) for r in piores_candidatas]
    
    pior_rot_global = piores_candidatas[np.argmax(amplitudes)]
    
    # 3. Define o pior_desvio_3D que o seu relatório está pedindo
    pior_desvio_3D = pior_rot_global[2,0]*Raio_Datum_A*np.cos(rads_ref) + \
                     pior_rot_global[2,1]*Raio_Datum_A*np.sin(rads_ref)

    # --- RELATÓRIO FINAL COM FORMATO DE TOLERÂNCIA ---
    num_falhas = sum(1 for r in resultados_compressao if r > 0)
    Falhas = (num_falhas / iteracoes) * 1000000.0
    
    # Cálculo do CPK Real baseado nas falhas
    cpk_real = abs(stats.norm.ppf(num_falhas / iteracoes))/3 if num_falhas > 0 else 2.0
    
    # Range estatístico (Percentis 0.003% e 99.997% -> Equivalente a +/- 4 Sigma / CPK 1.33)
    lim_inf = np.percentile(gaps_extremos_bimodal, 0.003)
    lim_sup = np.percentile(gaps_extremos_bimodal, 99.997)
    
    # Calcula a variação simétrica em relação ao nominal para o formato +-
    variacao_estatistica = (lim_sup - lim_inf) / 2
    pior_gap_encontrado = Gap_nom + np.max(np.abs(pior_desvio_3D))

    print(f"\n{'='*45}")
    print(f"          REPORT")
    print(f"{'='*45}")
    print(f"Simulated parts:             {iteracoes:,}")
    print(f"Nominal gap:                 {Gap_nom:.2f} mm")
    print(f"Estimated range (CPK 1.33):   {Gap_nom:.2f} ± {variacao_estatistica:.3f} mm")
    print(f"   > Lower limit:        {lim_inf:.3f} mm")
    print(f"   > Upper limit:        {lim_sup:.3f} mm")
    print(f"worst simulated gap:           {pior_gap_encontrado:.3f} mm")
    print(f"Espected failures:            {Falhas:.0f} PPM")
    print(f"CPK:         {cpk_real:.2f}")
    print(f"{'='*45}")
