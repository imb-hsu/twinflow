### Pallet configurations

The scenario generator supports three main pallet configurations used to create products in the system.

#### `PalletConfiguration-NOVA-6-DTF-6`

This configuration contains 12 products in total: six `NOVA` products and six `DTF` products. The products are arranged over two height levels. `NOVA` products are placed on the first level, while `DTF` products are placed on the second level.

Production routes:
- `NOVA` products visit `RS_1` and `RS_3`.
- `DTF` products visit `RS_2` and `RS_4`.

#### `PalletConfiguration-NOVA-DTF-HSU-Mix`

This configuration contains a mixed pallet with `NOVA`, `DTF`, and `HSU` products. It is used to represent a more heterogeneous production scenario in which different product types with different processing requirements are created on the same pallet.

Production routes:
- `NOVA` products visit `RS_1` and `RS_3`.
- `DTF` products visit `RS_2` and `RS_4`.
- `HSU` products visit `RS_1`, `RS_2`, `RS_2`, `RS_4`, `RS_5`, and `RS_6`.

#### `HSU-1Box`

This configuration contains a single `HSU` product. It represents a simple pallet setup with only one product and is useful for testing individual product routing and processing behavior.

Production route:
- `HSU` products visit `RS_1`, `RS_2`, `RS_2`, `RS_4`, `RS_5`, and `RS_6`.