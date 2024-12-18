. config
# rm -f E.npy
# python user/understand/twisted.py fields first x model 19.5
# rm -f E.npy
# python user/understand/twisted.py fields both y model 19.5
# rm -f E.npy
# python user/understand/twisted.py fields both y opt data/ellipsis_largepop_12_16_pso/free_pixmap_77/best.npz 19.5
rm -f figs/fields/*.p{ng,df}
for i in {0..15}; do
    echo "ta = $i"
    rm -f E.npy
    python user/understand/twisted.py fields both y opt data/ellipsis_largepop_12_16_pso/free_pixmap_77/best.npz $((4*$i))
done


for i in {0..15}; do
    mv figs/fields/efield_both_y_opt_$((4*$i)).0.png figs/fields/field-$(($i+1)).png
done
zip  field.zip figs/fields/field-*.png