for file in *.obj
do
    cmd="meshlabserver -i $file -o ${file%.*}_.dae -s convMeshes.mlx -om vc vn wt"
    echo $cmd
    $cmd
done

