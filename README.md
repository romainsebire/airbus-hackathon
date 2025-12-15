# hackathon-airbus

# Hackathon – Prédiction de KPIs Industriels

## 1. Évaluation

### Modalités

* **Rapport écrit** : 5 à 6 pages maximum
* **Soutenance** : 25 minutes

  * 10 minutes de présentation
  * 15 minutes de questions
* **Date de soutenance** : 19 décembre 2025
* **Date limite de soumission (codes + rapport)** : 18 décembre 2025 à 18h, sur campus

### Attendus pour le rapport

Le rapport doit :

* Décrire rapidement le **sujet**
* Décrire rapidement les **données**
* Se concentrer principalement sur **l’approche proposée** et les **résultats obtenus**

---

## 2. Contexte industriel

Dans l’industrie aéronautique et manufacturière, la gestion des ateliers est un enjeu stratégique. Les managers doivent configurer efficacement les ressources (opérateurs, lignes de production, investissements) afin de respecter les délais de livraison, tout en garantissant un haut niveau de satisfaction client.

La **simulation industrielle** permet d’explorer différents scénarios de production. Cependant, son principal inconvénient réside dans son **temps de calcul élevé**, ce qui limite son utilisation opérationnelle à grande échelle.

Ce hackathon propose une **approche data-driven**, fondée sur des modèles prédictifs capables d’approximer les résultats de simulations industrielles à partir :

* des demandes clients,
* des paramètres de configuration de l’atelier.

---

## 3. Objectif du hackathon

L’objectif principal est de **prédire simultanément trois indicateurs clés de performance (KPIs)** :

* **Customer Satisfaction** : proportion des pièces d’hélicoptère livrées à temps
* **Work In Progress (WIP)** : niveau d’encours de production
* **Investment** : investissements nécessaires pour la configuration de l’atelier

Le problème est formulé comme un **problème de régression multi-sorties**, permettant de capturer les dépendances et compromis entre ces indicateurs.

### Critère d’évaluation principal

* Seule la **customer satisfaction** est utilisée pour établir le score du leaderboard.
* Les indicateurs **WIP** et **investment** ne sont pas inclus dans le score final, mais :

  * ils sont nécessaires à l’entraînement du modèle,
  * ils permettent d’analyser les compromis industriels,
  * ils enrichissent l’interprétation des résultats.

 **la satisfaction client est prioritaire**.


Les sections suivantes du rapport détaillent :

* la méthodologie adoptée,
* les choix de modèles,
* les résultats obtenus,
* et les enseignements industriels associés.
