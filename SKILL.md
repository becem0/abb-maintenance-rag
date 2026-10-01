# Skill: ABB ACS880 Industrial Diagnostic & Maintenance RAG
**Specification Version:** 1.0  
**Domain:** Industrial Automation & Electrical Engineering (ABB Drives)  
**Evaluator Target:** NVIDIA SkillEvaluator Tier 1 / Tier 2 / Tier 3  

## 1. Description
Ce Skill permet à un agent IA d'assister un technicien de maintenance électrique dans le diagnostic, 
le câblage et la réparation des variateurs industriels ABB ACS880 à partir des manuels constructeur 
Hardware et Firmware.

## 2. Capabilities & Quality Gates
- **Cross-Lingual Technical Query Expansion:** Traduction contextuelle de requêtes FR vers l'anglais technique constructeur.
- **Root-Cause Analysis:** Arbre de défaillance dichotomique par questions fermées (OUI / NON).
- **Strict Grounding & Page Citations:** Citation systématique des pages constructeur ABB.
- **Safety & Abstention Guardrail:** Abstention obligatoire en cas d'absence d'information technique pour éviter les accidents électriques.

## 3. Evaluation Benchmark (NVIDIA Tier 3 Spec)
- **Baseline Arm:** Retrieval direct sans expansion ni garde-fous d'abstention.
- **With-Skill Arm:** Pipeline complet avec reformulation bilingue, raisonnement 120B et abstention stricte.
- **Primary Metric:** Skill Lift (Gain net de performance en Exactitude, Sécurité et Fidélité).