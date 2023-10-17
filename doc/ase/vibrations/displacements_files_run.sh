mkdir ethanol_forces

for DISP in ethanol_disp/*.xyz
do
    echo $DISP
    gpaw -P 4 run -p mode=lcao,basis=dzp $DISP -o ethanol_forces/$(basename $DISP)
done
