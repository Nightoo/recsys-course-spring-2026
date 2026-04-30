## Homework 2 Report

## Abstract

MDP-based Recommender

## Детали реализации

логи делятся на сессии по 30 минут. строятся переходы между треками с весами по частоте и длительности. итеративно решается уравнение беллмана (γ=0.9). скор перехода: P*(T+γ·V). для текущего трека выдаётся 100 следующих 

## AB
treatment                  metric  effect_pct  upper_pct  lower_pct  control_mean  treatment_mean  significant
       T1                    time        3.27       6.16       0.38       21.4941         22.1964         True
       T1                sessions        0.32       2.41      -1.78        3.1529          3.1629        False
       T1 mean_tracks_per_session        1.90       3.13       0.68       11.8959         12.1224         True
       T1   mean_time_per_session        3.41       5.37       1.44        6.9019          7.1371         True
       T1    mean_request_latency        0.37       0.54       0.20        0.7918          0.7948         True

