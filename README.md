# ingress_fssolver

文件夹`obj`和`sub`自己新建

前序准备：`scraper.py`（自己做）→`downloader.py`

出题后：`spliter.py`→`solver.py`

## mysql
```
CREATE TABLE `ingress_portals` (
  `external_id` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
  `name` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
  `url` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
  `lat` varchar(255) DEFAULT NULL,
  `lon` varchar(255) DEFAULT NULL,
  `updated` datetime DEFAULT NULL,
  `imported` datetime DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
```
