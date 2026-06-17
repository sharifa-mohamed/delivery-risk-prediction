import numpy as np

class LateDeliveryPreprocessingPipeline:
    def __init__(self, artifacts: dict):
        # Store mappings and defaults
        self.imputation_values = artifacts["imputation_values"]
        self.high_urgency_modes = artifacts["high_urgency_modes"]
        self.dropped_columns = artifacts["dropped_columns"]

        self.regional_volume_shares = artifacts["regional_volume_shares"]
        self.global_region_mean = artifacts["global_region_mean"]

        self.cust_counts_map = artifacts["cust_counts_map"]
        self.cust_sums_map = artifacts["cust_sums_map"]
        self.cust_sales_map = artifacts["cust_sales_map"]
        self.cust_disc_map = artifacts["cust_disc_map"]
        self.global_cust_count = artifacts["global_cust_count"]
        self.global_cust_sum = artifacts["global_cust_sum"]
        self.global_cust_sales = artifacts["global_cust_sales"]
        self.global_cust_disc = artifacts["global_cust_disc"]

        self.prod_counts_map = artifacts["prod_counts_map"]
        self.prod_sales_map = artifacts["prod_sales_map"]
        self.prod_disc_map = artifacts["prod_disc_map"]
        self.global_prod_count = artifacts["global_prod_count"]
        self.global_prod_sales = artifacts["global_prod_sales"]
        self.global_prod_disc = artifacts["global_prod_disc"]

        self.region_counts_map = artifacts["region_counts_map"]
        self.region_sales_map = artifacts["region_sales_map"]
        self.region_disc_map = artifacts["region_disc_map"]
        self.global_region_count = artifacts["global_region_count"]
        self.global_region_sales = artifacts["global_region_sales"]
        self.global_region_disc = artifacts["global_region_disc"]

        self.dept_sales_map = artifacts["dept_sales_map"]
        self.dept_counts_map = artifacts["dept_counts_map"]
        self.dept_disc_map = artifacts["dept_disc_map"]
        self.global_dept_sales = artifacts["global_dept_sales"]
        self.global_dept_count = artifacts["global_dept_count"]
        self.global_dept_disc = artifacts["global_dept_disc"]

        self.outlier_bounds = artifacts["outlier_bounds"]
        self.structural_cols = artifacts["structural_cols"]

        # Models/Transformers
        self.encoder = artifacts["encoder"]
        self.scaler = artifacts["scaler"]

        # Features lists to guarantee strict columns ordering
        self.vif_selected_columns = artifacts["vif_selected_columns"]

        self.scaling_columns = artifacts["scaling_columns"]

        self.lr_features = artifacts["selected_lr_columns"]
        self.tree_features = artifacts["tree_columns"]

    def _feature_engineering(self, df):
        df = df.copy()

        # Shipping Pressure
        df["shipping_pressure_index"] = np.where(
            df["Days for shipment (scheduled)"] == 0,
            df["Order Item Quantity"],
            df["Order Item Quantity"] / df["Days for shipment (scheduled)"]
        )

        # Flags
        df["is_high_urgency_mode"] = df["Shipping Mode"].isin(self.high_urgency_modes).astype(int)
        df["is_bulk_order"] = (df["Order Item Quantity"] > 3).astype(int)

        # Complexity & stress
        df["order_complexity_score"] = (df["Order Item Quantity"] * df["Order Item Product Price"]) * (1 - df["Order Item Discount Rate"])
        df["discount_per_item"] = df["Order Item Discount"] / (df["Order Item Quantity"] + 1)
        df["high_value_order"] = df["Order Item Product Price"] * df["Order Item Quantity"]

        # Regional congestion score fallback mapping
        df["regional_congestion_score"] = df["Order Region"].map(self.regional_volume_shares).fillna(self.global_region_mean)

        return df

    def _map_features(self, df):
        df = df.copy()

        # Customer
        df["customer_order_count"] = df["Order Customer Id"].map(self.cust_counts_map).fillna(self.global_cust_count)
        df["customer_total_quantity"] = df["Order Customer Id"].map(self.cust_sums_map).fillna(self.global_cust_sum)
        df["customer_avg_order_value"] = df["Order Customer Id"].map(self.cust_sales_map).fillna(self.global_cust_sales)
        df["customer_avg_discount"] = df["Order Customer Id"].map(self.cust_disc_map).fillna(self.global_cust_disc)

        # Product
        df["product_order_frequency"] = df["Product Name"].map(self.prod_counts_map).fillna(self.global_prod_count)
        df["product_avg_sales"] = df["Product Name"].map(self.prod_sales_map).fillna(self.global_prod_sales)
        df["product_avg_discount"] = df["Product Name"].map(self.prod_disc_map).fillna(self.global_prod_disc)

        # Region
        df["region_order_volume"] = df["Order Region"].map(self.region_counts_map).fillna(self.global_region_count)
        df["region_avg_sales"] = df["Order Region"].map(self.region_sales_map).fillna(self.global_region_sales)
        df["region_avg_discount"] = df["Order Region"].map(self.region_disc_map).fillna(self.global_region_disc)

        # Department
        df["department_avg_sales"] = df["Department Name"].map(self.dept_sales_map).fillna(self.global_dept_sales)
        df["department_order_volume"] = df["Department Name"].map(self.dept_counts_map).fillna(self.global_dept_count)
        df["department_avg_discount"] = df["Department Name"].map(self.dept_disc_map).fillna(self.global_dept_disc)

        return df

    def _clip_outliers(self, df):
        df = df.copy()
        for col, (low, high) in self.outlier_bounds.items():
            if col in df.columns:
                df[col] = df[col].clip(low, high)
        return df

 
    def transform(self, df):
        df = df.copy()

        df = df.drop(
            columns=self.dropped_columns,
            errors="ignore"
        )

        df = df.fillna(self.imputation_values)

        df = self._feature_engineering(df)
        df = self._map_features(df)

        df = df.drop(columns=self.structural_cols, errors="ignore")

        print("before transform shape:", df.shape)

        df = self.encoder.transform(df)

        print("after transform shape:", df.shape)

        df = self._clip_outliers(df)

        return df


    def transform_lr(self, df):
        df = self.transform(df)

        # Scale using the identical 26 features seen during training fit
        df[self.scaling_columns] = self.scaler.transform(df[self.scaling_columns])

        # Drop the VIF multi-collinear features safely here via reindexing
        df = df.reindex(columns=self.lr_features, fill_value=0)
        return df

    def transform_tree(self, df):
        df = self.transform(df)
        return df[self.tree_features]



