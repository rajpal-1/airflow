/*!
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

import React, { useMemo, useState } from 'react';
import {
  Flex,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  chakra,
  Alert,
  AlertIcon,
  Progress,
  Switch,
  IconButton,
  Text,
  useColorModeValue,
} from '@chakra-ui/react';
import {
  useTable, useSortBy, Column, usePagination,
} from 'react-table';
import {
  MdArrowDropDown, MdArrowDropUp, MdPlayArrow, MdKeyboardArrowLeft, MdKeyboardArrowRight,
} from 'react-icons/md';

import { defaultDags } from 'api/defaults';
import { useDags } from 'api';
import {
  DagName, PauseToggle, TriggerDagButton, DagTag,
} from './Row';

const getRandomInt = (max: number) => Math.floor(Math.random() * max);

// Generate 1-10 fake rows to show a skeleton loader
const skeletonLoader = [...Array(getRandomInt(10) || 1)].map(() => ({
  isPaused: <Switch disabled />,
  tags: '',
  dagId: <Progress size="lg" isIndeterminate data-testid="pipelines-loading" />,
  trigger: <IconButton size="sm" icon={<MdPlayArrow />} aria-label="Trigger Dag" disabled />,
}));

const LIMIT = 25;

const PipelinesTable: React.FC = () => {
  const [offset, setOffset] = useState(0);
  const {
    data: { dags, totalEntries } = defaultDags,
    isLoading,
    error,
  } = useDags({ limit: LIMIT, offset });

  const oddColor = useColorModeValue('gray.50', 'gray.900');
  const hoverColor = useColorModeValue('gray.100', 'gray.700');

  const pageCount = Math.ceil(totalEntries / LIMIT) || 1;

  const data = useMemo(
    () => (isLoading && !dags.length
      ? skeletonLoader
      : dags.map((d) => ({
        ...d,
        tags: d.tags.map((tag) => <DagTag tag={tag} key={tag.name} />),
        dagId: <DagName dagId={d.dagId} />,
        trigger: <TriggerDagButton dagId={d.dagId} />,
        active: <PauseToggle dagId={d.dagId} isPaused={d.isPaused} offset={offset} />,
      }))),
    [dags, isLoading, offset],
  );

  const columns = useMemo<Column<any>[]>(
    () => [
      {
        Header: 'Active',
        accessor: 'active',
        sortType: (rowA, rowB) => (rowA.original.isPaused && !rowB.original.isPaused ? 1 : -1),
      },
      {
        Header: 'Dag Id',
        accessor: 'dagId',
      },
      {
        Header: 'Tags',
        accessor: 'tags',
      },
      {
        disableSortBy: true,
        accessor: 'trigger',
      },
    ],
    [],
  );

  const {
    getTableProps,
    getTableBodyProps,
    allColumns,
    prepareRow,
    page,
    canPreviousPage,
    canNextPage,
    nextPage,
    previousPage,
    state: { pageIndex },
  } = useTable(
    {
      columns,
      data,
      pageCount,
      manualPagination: true,
      initialState: { pageIndex: offset / LIMIT, pageSize: LIMIT },
    },
    useSortBy,
    usePagination,
  );

  const handleNext = () => {
    nextPage();
    setOffset((pageIndex + 1) * LIMIT);
  };

  const handlePrevious = () => {
    previousPage();
    setOffset((pageIndex - 1 || 0) * LIMIT);
  };

  return (
    <>
      {error && (
      <Alert status="error" my="4" key={error.message}>
        <AlertIcon />
        {error.message}
      </Alert>
      )}
      <Table {...getTableProps()}>
        <Thead>
          <Tr>
            {allColumns.map((column) => (
              <Th
                {...column.getHeaderProps(column.getSortByToggleProps())}
              >
                {column.render('Header')}
                <chakra.span pl="2">
                  {column.isSorted && (
                    column.isSortedDesc ? (
                      <MdArrowDropDown aria-label="sorted descending" style={{ display: 'inline' }} size="2em" />
                    ) : (
                      <MdArrowDropUp aria-label="sorted ascending" style={{ display: 'inline' }} size="2em" />
                    )
                  )}
                </chakra.span>
              </Th>

            ))}
          </Tr>
        </Thead>
        <Tbody {...getTableBodyProps()}>
          {(!isLoading && !dags.length) && (
          <Tr>
            <Td colSpan={2}>No Pipelines found.</Td>
          </Tr>
          )}
          {page.map((row) => {
            prepareRow(row);
            return (
              <Tr
                {...row.getRowProps()}
                _odd={{ backgroundColor: oddColor }}
                _hover={{ backgroundColor: hoverColor }}
              >
                {row.cells.map((cell) => (
                  <Td
                    {...cell.getCellProps()}
                    py={3}
                  >
                    {cell.render('Cell')}
                  </Td>
                ))}
              </Tr>
            );
          })}
        </Tbody>
      </Table>
      <Flex alignItems="center" justifyContent="flex-end">
        <IconButton variant="ghost" onClick={handlePrevious} disabled={!canPreviousPage} aria-label="Previous Page">
          <MdKeyboardArrowLeft />
        </IconButton>
        <IconButton variant="ghost" onClick={handleNext} disabled={!canNextPage} aria-label="Next Page">
          <MdKeyboardArrowRight />
        </IconButton>
        <Text>
          {pageIndex + 1}
          {' of '}
          {pageCount}
        </Text>
      </Flex>
    </>
  );
};

export default PipelinesTable;
