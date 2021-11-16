# Dutch Data Warehouse (DDW)

The real-data experiment in the paper uses the [Dutch Data Warehouse](https://pubmed.ncbi.nlm.nih.gov/34425864/), which is protected by a license and cannot be shared here.
If you are interested in the dataset, please contact the co-author [Paul Elbers](https://nl.linkedin.com/in/paulelbersmdphd) for more information. 

*After* you have obtained access, you will need access to `data_warehouse_utils` utility module, contact [Zhaozhi Qian](http://www.damtp.cam.ac.uk/person/zq224).



## Preprocessing

To preprocess the data:
* Place the data files under [data/](../data/).
* Make sure `data_warehouse_utils` is in your Python path.
* Run preprocessing scripts (from repo root):
    ```bash
    $ python -m real_data.temporal_feat
    $ python -m real_data.temporal_feat_process
    $ python -m real_data.temporal_treatment
    ```
