#!/usr/bin/env python3
"""
Logistic Regression from scratch - Luis Balderas A01751150
"""

import argparse, csv, math, random, sys
from typing import List

#intentamos convertir a númerico
def _can_float(s: str) -> bool:
    try:
        float(s)
        return True
    except:
        return False

def _build_column_types(rows, target_col):
    """
    Detecta columnas numericas vs categoricas
    Regresa dos listas: numeric_cols, categorical_cols
    """
    header = list(rows[0].keys())
    numeric_cols, categorical_cols = [], []
    for c in header:
        if c == target_col:
            continue
        # si todas las filas son convertibles a float => numerica; si alguna no, categórica
        all_float = True
        for r in rows:
            if not _can_float(r[c]):
                all_float = False
                break
        if all_float:
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)
    return numeric_cols, categorical_cols

def _fit_one_hot_maps(rows, categorical_cols):
    """
    Para cada columna categorica, junta el conjunto de categorias observadas
    y crea el mapeo de nombres de columnas one-hot resultantes
    """
    cat_values = {c: [] for c in categorical_cols}
    seen = {c: set() for c in categorical_cols}
    for r in rows:
        for c in categorical_cols:
            v = r[c]
            if v not in seen[c]:
                seen[c].add(v)
                cat_values[c].append(v)
    # genera nombres de columnas one-hot
    one_hot_cols = []
    for c in categorical_cols:
        for v in cat_values[c]:
            one_hot_cols.append(f"{c}__{v}")
    return cat_values, one_hot_cols

def _encode_row_numeric_and_one_hot(r, numeric_cols, categorical_cols, cat_values):

    xi = [float(r[c]) for c in numeric_cols]  # numericos primero
    for c in categorical_cols:
        vals = cat_values[c]
        val = r[c]
        for v in vals:
            xi.append(1.0 if val == v else 0.0)
    return xi

def _parse_target_to_int(v):
    try:
        return int(v)
    except:
        s = str(v).strip().lower()
        if s in ("1", "true", "yes", "y", "passed", "pass", "positive", "pos","p"):
            return 1
        if s in ("0", "false", "no", "n", "failed", "fail", "negative", "neg","e"):
            return 0
        # ultimo recurso: si es floatable, umbral 0.5
        if _can_float(s):
            return 1 if float(s) >= 0.5 else 0
        raise ValueError(f"No puedo interpretar el target '{v}' como 0/1.")
# ---------------------------------------------------------------------------

def read_csv_numeric_features(path: str, target_col: str):
    with open(path, newline='', encoding='utf-8') as f: ##leeemos el archivo
        reader = csv.DictReader(f) #lo convertimos a diccionario recordemos que actua como un generador
        rows = list(reader) #entonces hacemos que el generador itere en todo
    # --- NUEVO: detectar columnas numericas vs categoricas (excluyendo target)
    numeric_cols, categorical_cols = _build_column_types(rows, target_col)

    # Si no hay categoricas, mantenemos exactamente tu comportamiento original
    if len(categorical_cols) == 0:
        feature_cols = [c for c in rows[0].keys() if c != target_col] 
        X, y = [], []
        for r in rows:
            xi = [float(r[c]) for c in feature_cols] #guardamos las columnas con las que se va a usar la regresion
            yi = _parse_target_to_int(r[target_col])#la columna objetivo (clase)
            X.append(xi); y.append(yi)
        return X, y, feature_cols

    # Si hay categoricas, construimos one-hot y concatenamos con numericas
    cat_values, one_hot_cols = _fit_one_hot_maps(rows, categorical_cols)
    # El orden final de features: primero numericas, luego una columna por categoria
    feature_cols = numeric_cols + one_hot_cols

    X, y = [], []
    for r in rows:
        xi = _encode_row_numeric_and_one_hot(r, numeric_cols, categorical_cols, cat_values)
        yi = _parse_target_to_int(r[target_col])#la columna objetivo (clase)
        X.append(xi); y.append(yi)
    return X, y, feature_cols



#recibe la lista con nuestro features y nuestra columna objetivo
def train_test_split(X, y, test_size=0.3, seed=42):
    n = len(X) #filas en el dataset
    idx = list(range(n)) #creamos un indice 
    random.Random(seed).shuffle(idx) #mezclamos el indice al azar
    n_test = int(round(n*test_size)) #el test size lo definimos como 0.3 (30%)
    test_idx = set(idx[:n_test])
    X_train, y_train, X_test, y_test = [], [], [], []
    for i in range(n):
        if i in test_idx:#si el contador actual esta en el indice de test
            X_test.append(X[i]); y_test.append(y[i])
        else:
            X_train.append(X[i]); y_train.append(y[i])
    return X_train, y_train, X_test, y_test

def sigmoid(z):  # funcion sigmoide numericamente estable
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)


#hacemos el producto punto
def dot(a,b): return sum(x*y for x,y in zip(a,b))

class LogisticRegressionScratch:
    def __init__(self, n_features, lr=0.1, epochs=1000): #un lr bajo 
        self.lr = lr; self.epochs = epochs
        self.w = [0.0]*n_features
        self.b = 0.0
        self.loss_history = [] #aqui guardaremos si va mejorando la perdida
    def _loss(self, X, y):
        eps=1e-12
        n=len(X); total=0.0
        for i in range(n):
            z = dot(self.w,X[i])+self.b
            p = sigmoid(z)
            p = min(max(p,eps),1-eps)
            total += -(y[i]*math.log(p)+(1-y[i])*math.log(1-p))
        return total/n
    def fit(self,X,y):
        n=len(X); d=len(X[0])
        for epoch in range(self.epochs):
            grad_w=[0.0]*d; grad_b=0.0
            for i in range(n):
                z=dot(self.w,X[i])+self.b
                p=sigmoid(z)
                err=p-y[i]
                for j in range(d):
                    grad_w[j]+=err*X[i][j]
                grad_b+=err
            for j in range(d):
                self.w[j]-=self.lr*grad_w[j]/n
            self.b-=self.lr*grad_b/n
            self.loss_history.append(self._loss(X,y))
    def predict(self,X,threshold=0.5):
        return [1 if sigmoid(dot(self.w,xi)+self.b)>=threshold else 0 for xi in X]



def confusion_matrix(y_true,y_pred):
    tp=sum(1 for yt,yp in zip(y_true,y_pred) if yt==1 and yp==1)
    tn=sum(1 for yt,yp in zip(y_true,y_pred) if yt==0 and yp==0)
    fp=sum(1 for yt,yp in zip(y_true,y_pred) if yt==0 and yp==1)
    fn=sum(1 for yt,yp in zip(y_true,y_pred) if yt==1 and yp==0)
    return tp,tn,fp,fn

def metrics(tp,tn,fp,fn):
    total=tp+tn+fp+fn
    acc=(tp+tn)/total if total else 0
    prec=tp/(tp+fp) if tp+fp>0 else 0
    rec=tp/(tp+fn) if tp+fn>0 else 0
    f1=2*prec*rec/(prec+rec) if prec+rec>0 else 0
    return acc,prec,rec,f1

# ---- añadimos funcion para guardar los datasets ----
def save_dataset(path, X, y, feature_cols):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(feature_cols + ["target"])
        for xi, yi in zip(X, y):
            writer.writerow(list(xi) + [yi])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True)
    ap.add_argument("--target",required=True)
    ap.add_argument("--test-size",type=float,default=0.3)
    ap.add_argument("--lr",type=float,default=0.1)
    ap.add_argument("--epochs",type=int,default=1000)
    # argumentos para guardar los splits
    ap.add_argument("--train-out", default="train_dataset.csv")
    ap.add_argument("--test-out", default="test_dataset.csv")
    args=ap.parse_args()

    X,y,features=read_csv_numeric_features(args.data,args.target)
    X_train,y_train,X_test,y_test=train_test_split(X,y,test_size=args.test_size)

    # guardamos datasets
    save_dataset(args.train_out, X_train, y_train, features)
    save_dataset(args.test_out, X_test, y_test, features)

    model=LogisticRegressionScratch(len(features),lr=args.lr,epochs=args.epochs)
    model.fit(X_train,y_train)

    y_pred=model.predict(X_test)
    tp,tn,fp,fn=confusion_matrix(y_test,y_pred)
    acc,prec,rec,f1=metrics(tp,tn,fp,fn)

    print("\n" + "="*50)
    print("      Logistic Regression (basic) - Resultados")
    print("="*50)
    print(f"Features usados ({len(features)}):")
    for f, w in zip(features, model.w):
        print(f"  {f:<20} -> {w:.4f}")
    print(f"Bias: {model.b:.4f}")

    print("\nMatriz de confusión (y_true filas, y_pred columnas):")
    print(f"          Pred 0    Pred 1")
    print(f"True 0     {tn:5d}    {fp:5d}")
    print(f"True 1     {fn:5d}    {tp:5d}")

    print("\nMétricas de desempeño:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-score : {f1:.4f}")

    print("\nArchivos guardados:")
    print(f"  Train -> {args.train_out}")
    print(f"  Test  -> {args.test_out}")
    print("="*50 + "\n")


if __name__=="__main__":
    main()
