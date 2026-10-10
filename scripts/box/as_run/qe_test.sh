#!/bin/bash
# Smoke test of the conda-forge QE build: 5-atom cubic BaTiO3 SCF, SSSP 1.3 efficiency.
export PATH=/root/qe/bin:$PATH
export LD_LIBRARY_PATH=/root/qe/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
mkdir -p /root/qetest && cd /root/qetest
cat > pw.in <<'EOF'
&control
  calculation='scf', prefix='bto', outdir='./tmp', pseudo_dir='/root/sssp/upf',
  tprnfor=.true., disk_io='none'
/
&system
  ibrav=0, nat=5, ntyp=3, ecutwfc=50, ecutrho=400,
  occupations='smearing', smearing='gaussian', degauss=0.005
/
&electrons
  conv_thr=1e-9, mixing_beta=0.4
/
ATOMIC_SPECIES
Ba 137.327 Ba.pbe-spn-kjpaw_psl.1.0.0.UPF
Ti 47.867 ti_pbe_v1.4.uspp.F.UPF
O 15.999 O.pbe-n-kjpaw_psl.0.1.UPF
CELL_PARAMETERS angstrom
4.0 0.0 0.0
0.0 4.0 0.0
0.0 0.0 4.0
ATOMIC_POSITIONS crystal
Ba 0.0 0.0 0.0
Ti 0.5 0.5 0.52
O 0.5 0.5 0.0
O 0.5 0.0 0.5
O 0.0 0.5 0.5
K_POINTS automatic
6 6 6 0 0 0
EOF
mpirun --allow-run-as-root -np 16 pw.x -nk 4 -in pw.in > pw.out 2> time.log < /dev/null
grep -E "^!|Total force|JOB DONE|convergence has been achieved" pw.out
cat time.log | tail -1
pw.x -h 2>/dev/null | head -0
grep "Program PWSCF" pw.out
