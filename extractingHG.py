import numpy as np
import pandas as pd
import networkx as nx 
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# from torch_geometric.utils.convert import from_networkx
# from sklearn.model_selection import train_test_split
# import os
# import torch
# import torch.nn as nn
# from scipy.optimize import minimize


def getDict(A, DF, label1, label2):
    # Extracting the list of NODES for each A item. 
    # Filter the dataframe to extract Nodes associated with the specified label1
    # Extract the list of Nodes associated with the patient
    D = {}
    for v in A:
        df = DF[DF[label1] == v]
        id_list = df[label2].tolist()
        if len(id_list)>0:
            D[v] = id_list
    
    return D


def getNodes_and_Edges(D):
    DEdges = []
    DNodes = []
    for i, v in D.items():
        for j in v:
            if isinstance(j, (int, float)) and not np.isnan(j):
                DEdges.append([i, j])
                if j not in DNodes:
                    DNodes.append(j)
            elif isinstance(j, str):
                # Check if it's a valid numeric string before treating it as a node
                try:
                    numeric_value = float(j)
                    DEdges.append([i, numeric_value])
                    if numeric_value not in DNodes:
                        DNodes.append(numeric_value)
                except ValueError:
                    # If it's not a valid numeric string, treat it as a regular string node
                    DEdges.append([i, j])
                    if j not in DNodes:
                        DNodes.append(j)
    return DNodes, remove_red(DEdges)

def remove_red(DEdges):
    f = []
    for u,v in DEdges:
        if [u,v] not in f:
            f.append([u,v])
    return f


def get_VXV(VX_edges, Nodes):
    N = len(Nodes)
    VXV_A = np.zeros((N, N))

    newG = nx.Graph()
    newG.add_edges_from(VX_edges)
    VNodes = [i for i in newG.nodes() if i[0]=="V"]
    for i in range(len(VNodes)-1):
        u = VNodes[i]
        Nu = set(newG.neighbors(u))
        for j in range(i+1, len(VNodes)):
            v = VNodes[j]
            Nv = set(newG.neighbors(v))
            ni, nj = Nodes.index(v), Nodes.index(u)

            Intersection = Nu.intersection(Nv)
            VXV_A[ni, nj] = 2 * (len(Intersection)) / (len(Nu) + len(Nv))
            VXV_A[nj, ni] = 2 * (len(Intersection)) / (len(Nu) + len(Nv))
    return VXV_A



def threeNodes(VA_edges, VB_edges, A_Nodes, B_Nodes, Nodes):
    N = len(Nodes)
    A = np.zeros((N, N))

    newG = nx.Graph()
    newG.add_edges_from(VA_edges)
    newG.add_edges_from(VB_edges)

    for u in A_Nodes:
        Nu = set(newG.neighbors(u))
        for v in B_Nodes:
            Nv = set(newG.neighbors(v))

            Intersection = Nu.intersection(Nv)
            w = 2 * (len(Intersection)) / (len(Nu) + len(Nv))
            ux, vx = Nodes.index(u), Nodes.index(v)
            A[ux, vx] = w
            A[vx, ux] = w
            # if w!=0:
            #     print(U, V, Intersection, w)
    return A


def heatmap(A, caption):
    cmap = LinearSegmentedColormap.from_list('custom_cmap', [(0, 'white'), (0.5, 'blue'), (1, 'red')])
    # plt.figure(figsize=(8, 8))  # Adjust the figsize as needed

    plt.imshow(A, cmap=cmap, interpolation='nearest')
    plt.colorbar()  # Add a colorbar to the heatmap
    plt.title(f'Heatmap for {caption}')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # Show the heatmap
    plt.show()    


# ----------------------------------------------------------------------------
folder_path = 'Data/MIMIC resources'

df_Admissions = pd.read_csv(f'{folder_path}/ADMISSIONS.csv')

df_Patients = pd.read_csv(f'{folder_path}/PATIENTS.csv')

# medication!
df_Prescription = pd.read_csv(f'{folder_path}/PRESCRIPTIONS.csv')

# Diagnosis!
df_DiagnosisICD = pd.read_csv(f'{folder_path}/DIAGNOSES_ICD.csv')

# Procedures!
df_ProceduresICD = pd.read_csv(f'{folder_path}/PROCEDURES_ICD.csv')
# ICUStays
df_Icustays = pd.read_csv(f'{folder_path}/ICUSTAYS.csv')


df_ProceduresICD.dropna(subset=['ICD9_CODE'], inplace=True)
df_Prescription.dropna(subset=['drug'], inplace=True)
df_DiagnosisICD.dropna(subset=['ICD9_CODE'], inplace=True)




# ----------------------------------------------------------------------------
# convert all procedure codes into two digits only
# convert all diagnosis codes into three digits only

def extract3(code):
    return str(code)[:3]
def extract2(code):
    return str(code)[:2]

df_DiagnosisICD['ICD9_CODE'] = df_DiagnosisICD['ICD9_CODE'].apply(extract3)
df_ProceduresICD['ICD9_CODE'] = df_ProceduresICD['ICD9_CODE'].apply(extract2)

df_ProceduresICD



# ----------------------------------------------------------------------------

Procedures = sorted(df_ProceduresICD['ICD9_CODE'].unique())
Medication = sorted(df_Prescription['drug'].unique())
Diagnosis  = df_DiagnosisICD['ICD9_CODE'].unique()
Patients = df_Patients['SUBJECT_ID'].unique()
Admissions = df_Admissions['HADM_ID'].unique()

print(f'Number of Patients = {len(Patients)}')

print(f'Number of Admissions = {len(Admissions)}')

print(f'Number of Diagnosis = {len(Diagnosis)}')

print(f'Number of procedures = {len(Procedures)}')

print(f'Number of Medication = {len(Medication)}')
# ----------------------------------------------------------------------------
# # restricting to LUNG disease
# ICD_Diagnosis_Lung = [i for i in Diagnosis if str(i).startswith('162')]

df_sub = df_DiagnosisICD[df_DiagnosisICD['ICD9_CODE'].str.startswith('162')]

df_sub = df_sub[['SUBJECT_ID',	'HADM_ID']].dropna(subset=['HADM_ID'])
# ----------------------------------------------------------------------------
Patients = df_sub['SUBJECT_ID'].unique()
Admissions = df_sub['HADM_ID'].unique()

print(f'Number of Patients = {len(Patients)}')
print(f'Number of Admission = {len(Admissions)}')
# ----------------------------------------------------------------------------
newDF = df_Admissions[df_Admissions['HADM_ID'].isin(Admissions)]
newDF = newDF[['SUBJECT_ID','HADM_ID','ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'HOSPITAL_EXPIRE_FLAG']]

newDF['ADMITTIME'] = pd.to_datetime(newDF['ADMITTIME'])
newDF['DISCHTIME'] = pd.to_datetime(newDF['DISCHTIME'])

newDF['LOS'] = (newDF['DISCHTIME']-newDF['ADMITTIME']).dt.days
# ----------------------------------------------------------------------------
df = newDF
df['ADMITTIME'] = pd.to_datetime(df['ADMITTIME'])
df['DEATHTIME'] = pd.to_datetime(df['DEATHTIME'])

# Filter rows with the least ADMITTIME per SUBJECT_ID
admit_df = df.sort_values('ADMITTIME').groupby('SUBJECT_ID').head(1)

admit_df = admit_df[['SUBJECT_ID','ADMITTIME']]

death_df = df[df['DEATHTIME'].notnull()].sort_values('DEATHTIME').groupby('SUBJECT_ID').head(1)

death_df = death_df[['SUBJECT_ID','DEATHTIME']]

final_df = admit_df.merge(death_df, on='SUBJECT_ID', how='outer').sort_values(by='SUBJECT_ID')

Patients = final_df['SUBJECT_ID'].unique()

final_df.to_csv('Data/survival.csv')
# ----------------------------------------------------------------------------
# Filter rows with non-null DEATHTIME and extract SUBJECT_ID values
dead = final_df[final_df['DEATHTIME'].notnull()]['SUBJECT_ID'].tolist()
dead_patients = [f'C_{i}' for i in dead]
# ----------------------------------------------------------------------------
# Edge Extractions
VisitDict      = getDict(Patients, df_sub, 'SUBJECT_ID', 'HADM_ID')

VisitNodes, PatientVisit = getNodes_and_Edges(VisitDict)
print(f'Total number of patient-visit = {len(PatientVisit)}')

DiagnosisDict  = getDict(VisitNodes, df_DiagnosisICD, 'HADM_ID', 'ICD9_CODE')
DiagnosisNodes, VisitDiagnosis = getNodes_and_Edges(DiagnosisDict)
print(f'Total number of Visit-Diagnosis = {len(VisitDiagnosis)}')

ProcedureDict  = getDict(VisitNodes, df_ProceduresICD, 'HADM_ID', 'ICD9_CODE')
ProcedureNodes, VisitProcedure = getNodes_and_Edges(ProcedureDict)
print(f'Total number of Visit-Procedure = {len(VisitProcedure)}')

MedicationDict = getDict(VisitNodes, df_Prescription, 'hadm_id', 'drug')
MedicationNodes, VisitMedication = getNodes_and_Edges(MedicationDict)
print(f'Total number of Visit-Medication = {len(VisitMedication)}')

ICUSTAYDict = getDict(VisitNodes, df_Prescription, 'hadm_id', 'icustay_id')
ICUSTAYNodes, VisitICUSTAY = getNodes_and_Edges(ICUSTAYDict)
print(f'Total number of Visit-ICUSTAY = {len(VisitICUSTAY)}')

# # -----------------------------------------------------------------------------
# ICUSTAY_MedicationDict = getDict(ICUSTAYNodes, df_Prescription, 'icustay_id', 'drug')
# _, ICUSTAY_Medication = getNodes_and_Edges(ICUSTAY_MedicationDict)
# print(f'Total number of ICUSTAY-Medication = {len(ICUSTAY_Medication)}')
# ----------------------------------------------------------------------------
# Mapping function for Nodes and edges
# # c, C : Patients
# v, V : visits
# d, D : Diagnosis
# p, P : Procedure
# m, M : Medication
# i, I : ICUSTAY

CV_edges = [[f'C_{u}', f'V_{v}'] for u,v in PatientVisit]
VD_edges = [[f'V_{u}', f'D_{v}'] for u,v in VisitDiagnosis]
VP_edges = [[f'V_{u}', f'P_{v}'] for u,v in VisitProcedure]
VM_edges = [[f'V_{u}', f'M_{v}'] for u,v in VisitMedication]
VI_edges = [[f'V_{u}', f'I_{v}'] for u,v in VisitICUSTAY]
# IM_edges = [[f'I_{u}', f'M_{v}'] for u,v in ICUSTAY_Medication]
# ----------------------------------------------------------------------------

edge_index = CV_edges + VD_edges + VP_edges + VM_edges #+ VI_edges

tempG = nx.Graph()
tempG.add_edges_from(edge_index)

Nodes = list(tempG.nodes())
N = len(Nodes)

C_Nodes = [v for v in Nodes if v[0]=='C']
V_Nodes = [v for v in Nodes if v[0]=='V']
M_Nodes = [v for v in Nodes if v[0]=='M']
D_Nodes = [v for v in Nodes if v[0]=='D']
P_Nodes = [v for v in Nodes if v[0]=='P']

CV = nx.Graph()
CV.add_nodes_from(C_Nodes)
CV.add_nodes_from(V_Nodes)
CV.add_edges_from(CV_edges)
nx.write_gml(CV, 'Data/gmls/CV.gml')

VM = nx.Graph()
VM.add_nodes_from(V_Nodes)
VM.add_nodes_from(M_Nodes)
VM.add_edges_from(VM_edges)
nx.write_gml(VM, 'Data/gmls/VM.gml')

VD = nx.Graph()
VD.add_nodes_from(V_Nodes)
VD.add_nodes_from(D_Nodes)
VD.add_edges_from(VD_edges)
nx.write_gml(VD, 'Data/gmls/VD.gml')

VR = nx.Graph()
VR.add_nodes_from(V_Nodes)
VR.add_nodes_from(P_Nodes)
VR.add_edges_from(VP_edges)
nx.write_gml(VR, 'Data/gmls/VP.gml')








# C_index = [i for i, v in enumerate(Nodes) if v[0]=='C']
# V_index = [i for i, v in enumerate(Nodes) if v[0]=='V']
# M_index = [i for i, v in enumerate(Nodes) if v[0]=='M']
# D_index = [i for i, v in enumerate(Nodes) if v[0]=='D']
# P_index = [i for i, v in enumerate(Nodes) if v[0]=='P']


# Nodes = C_Nodes + V_Nodes + M_Nodes + D_Nodes + P_Nodes
# # Hetereogenous graph
# HG = nx.Graph()
# HG.add_nodes_from(Nodes)
# HG.add_edges_from(edge_index)
# # ----------------------------------------------------------------------------

# # V-D-V

# VDV_A = np.zeros((N, N))
# VMV_A = np.zeros((N, N))
# VPV_A = np.zeros((N, N))

# VDV_A = get_VXV(VD_edges, N)
# VMV_A = get_VXV(VM_edges, N)
# VPV_A = get_VXV(VP_edges, N)
# # ----------------------------------------------------------------------------
# DVM_A = threeNodes(VD_edges, VM_edges, D_Nodes, M_Nodes, Nodes)
# CVM_A = threeNodes(CV_edges, VM_edges, C_Nodes, M_Nodes, Nodes)
# CVP_A = threeNodes(CV_edges, VP_edges, C_Nodes, P_Nodes, Nodes)
# CVD_A = threeNodes(CV_edges, VD_edges, C_Nodes, D_Nodes, Nodes)
# MVP_A = threeNodes(VM_edges, VP_edges, M_Nodes, P_Nodes, Nodes)
# DVP_A = threeNodes(VD_edges, VP_edges, D_Nodes, P_Nodes, Nodes)
# # ----------------------------------------------------------------------------
# A = np.eye(N, N)
# As1 = [A, VDV_A, DVM_A, CVM_A, CVP_A, CVD_A, MVP_A, DVP_A]
# As2 = [A, VMV_A, DVM_A, CVM_A, CVP_A, CVD_A, MVP_A, DVP_A]
# As3 = [A, VPV_A, DVM_A, CVM_A, CVP_A, CVD_A, MVP_A, DVP_A]

# A_meta1 = np.zeros((N,N))
# for a in As1:
#     A_meta1 +=a
# # heatmap(A_meta1, 'Visit-Diagnosis-Visit')

# A_meta2 = np.zeros((N,N))
# for a in As2:
#     A_meta2 +=a
# # heatmap(A_meta2, 'Visit-Medication-Visit')

# A_meta3 = np.zeros((N,N))
# for a in As3:
#     A_meta3 +=a
# # heatmap(A_meta3, 'Visit-Procedure-Visit')
# # ----------------------------------------------------------------------------
# # Lenght of stay (LOS)
# # ----------------------------------------------------------------------------
# Y = newDF[['HADM_ID', 'LOS']].set_index('HADM_ID')['LOS'].to_dict()
# # Extract degrees from the dictionary
# degrees = list(Y.values())


# # ----------------------------------------------------------------------------
# Nodes = list(HG.nodes())
# X = np.random.randn(len(Nodes), 128)
# # X = np.eye(Total_number_of_nodes)

# num_features = X.shape[1]
# G = None
# G = nx.Graph()
# for v in Nodes:
#     i = Nodes.index(v)
#     node_attributes = {'LOS': 99999}
#     if v[0]=='V':
#         # m = 1
#         # if Y[int(v[2:])]<=20:
#         #     m = 0
#         node_attributes = {'LOS': Y[int(v[2:])]}
#     if len(v)<2:
#         print(v)
#     G.add_node( v, **node_attributes)

# for i in range(len(Nodes)-1):
#     for j in range(i + 1, len(Nodes)):
#         G.add_edge(Nodes[i], Nodes[j], weight=A_meta1[i, j])
# # ----------------------------------------------------------------------------
# c, v, m, p, d = 0,0,0,0,0
# for n in Nodes:
#     if n[0]=='V':
#         v+=1
#     if n[0]=='C':
#         c+=1
#     if n[0]=='M':
#         m+=1
#     if n[0]=='P':
#         p+=1
#     if n[0]=='D':
#         d+=1

# print(f'Patients = {c}\nVisits = {v}\nMedications = {m}\nProcedures = {p}\nDiagnosis = {d}\nTotal = {c+v+m+d+p}')

# # ----------------------------------------------------------------------------
# nx.write_gml(G, 'Data/HGs/HG1.gml')
# # ----------------------------------------------------------------------------

# # ----------------------------------------------------------------------------

# # ----------------------------------------------------------------------------

# # ----------------------------------------------------------------------------

# # ----------------------------------------------------------------------------

# # def NodeType(v):
# #     return v[0]

# # def find_paths_of_length(G, source, target, length):
# #     all_paths = []
# #     for path in nx.all_simple_paths(G, source=source, target=target, cutoff=length):
# #         if len(path) == length + 1:  # Check if the path length matches the desired length
# #             all_paths.append(path)
# #     return all_paths

# # def PathCount(G, Nodes, metapath):
# #     print(f'Processing {metapath}')
# #     L = len(metapath)
# #     N = len(Nodes)
# #     A = np.zeros((N, N))

# #     start_nodes = [i for i in G.nodes() if NodeType(i)== metapath[0] ]
# #     end_nodes = [i for i in G.nodes() if NodeType(i)== metapath[-1] ]
# #     total = 0
# #     for n in start_nodes:
# #         for m in end_nodes:
# #             Paths = find_paths_of_length(G, n, m, L-1)
# #             count=0
# #             if len(Paths)>0:
# #                 for p in Paths:
# #                     mp = [NodeType(i) for i in p]
# #                     if mp==metapath:
# #                         count+=1
# #             A[Nodes.index(n), Nodes.index(m)]= count
# #             A[Nodes.index(m), Nodes.index(n)]= count
# #             total+=count
# #     print(total)
# #     return A


# # # VDV_A = PathCount(HG, Nodes, ['V', 'D','V'])
# # # VMV_A = PathCount(HG, Nodes, ['V', 'M','V'])
# # # VPV_A = PathCount(HG, Nodes, ['V', 'P','V'])

# # DVM_A = PathCount(HG, Nodes, ['D', 'V','M'])
# # CVM_A = PathCount(HG, Nodes, ['C', 'V','M'])
# # CVP_A = PathCount(HG, Nodes, ['C', 'V','P'])
# # CVD_A = PathCount(HG, Nodes, ['C', 'V','D'])
# # MVP_A = PathCount(HG, Nodes, ['M', 'V','P'])
# # DVP_A = PathCount(HG, Nodes, ['D', 'V','P'])

# # A = np.sum(DVM_A, CVM_A, CVP_A, CVD_A, MVP_A, DVP_A, VDV_A, VMV_A, VPV_A)
# # plt.imshow(A, cmap='viridis', interpolation='nearest')
# # plt.colorbar()  # Add a colorbar to the heatmap
# # plt.title('Heatmap Example')
# # plt.xlabel('X-axis')
# # plt.ylabel('Y-axis')

# # # Show the heatmap
# # plt.show()

