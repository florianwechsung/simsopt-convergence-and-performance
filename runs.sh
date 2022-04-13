for MODE in 0 1 2; do
    for OMP in 1 8 16 96; do
        export OMP_NUM_THREADS=$OMP
        python3 timing.py --MODE $MODE
    done
done
