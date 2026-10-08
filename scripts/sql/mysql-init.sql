-- Runs once when the demo MySQL container is first created.
CREATE USER 'support_ro'@'%' IDENTIFIED BY 'support_ro';
GRANT SELECT ON shop.* TO 'support_ro'@'%';
