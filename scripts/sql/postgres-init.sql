-- Runs once when the demo Postgres container is first created (as the container superuser).
-- The agent connects as `support_ro`, which can only read: it can never alter shop data.
CREATE ROLE support_ro LOGIN PASSWORD 'support_ro';
GRANT CONNECT ON DATABASE shop TO support_ro;
GRANT USAGE ON SCHEMA public TO support_ro;
-- Tables are created later by `support-agent seed-demo` (as shop_owner); make them readable.
ALTER DEFAULT PRIVILEGES FOR ROLE shop_owner IN SCHEMA public GRANT SELECT ON TABLES TO support_ro;
