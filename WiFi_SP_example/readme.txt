#--synced csi and image from rawdata---------------------------------
synceddata/collect_xxxxx/frame_*.npz

npz_data = np.load(npz_filename)
npz_data.f.rawcsi          #: complex ndarray# [sample/time= vary, width=3, height=3, carrier/frequency=30]
npz_data.f.rawcsistamp     #:ndarray of csi sample timestamp
npz_data.f.image           #: undistorted RGB ndarray # [768, 1024,3]
npz_data.f.depth           #: scene depth map ndarray # [768, 1024,1]
npz_data.f.segmentation    #: human segmenation map ndarray # [768, 1024,1]



